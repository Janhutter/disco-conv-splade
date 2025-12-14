"""
code for inverted index based on arrays, powered by numba based retrieval
"""

import array
import json
import os
import pickle
from collections import defaultdict

import h5py
import numpy as np
from tqdm.auto import tqdm


class IndexDictOfArray:
    """On-disk inverted index backed by HDF5.

    For large collections we want to avoid keeping all postings in memory.
    The original implementation accumulated everything in Python arrays and
    only wrote once at the end. Here we keep a small in-memory buffer per
    posting list and periodically flush append-only chunks to HDF5. This
    bounds memory while keeping the final on-disk format identical:

    - one HDF5 file ``array_index.h5py`` containing datasets
      ``index_doc_id_{k}`` and ``index_doc_value_{k}`` and a scalar
      ``dim``.
    - an ``index_dist.json`` describing posting-list lengths (used only
      for inspection).

    When ``force_new=False`` we load the full index into memory as before,
    so retrieval code relying on numpy arrays is unchanged.
    """

    def __init__(self, index_path=None, force_new=False, filename="array_index.h5py", dim_voc=None,
                 buffer_size: int = 100):
        self.buffer_size = int(buffer_size)
        self.index_path = index_path
        self.filename = None
        self.n = 0
        self._dim_voc = dim_voc

        # In-memory containers used both for buffered appends and for fully
        # loaded indexes (when force_new=False).
        self.index_doc_id = None
        self.index_doc_value = None

        if index_path is not None:
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            self.filename = os.path.join(index_path, filename)

        # When ``force_new=True`` we want a completely fresh index file.
        # If a previous run left a partial or corrupt ``array_index.h5py``
        # behind, appending to it can trigger low‑level HDF5 errors such as
        # "wrong B-tree signature". Removing the stale file here ensures
        # each indexing run starts from a clean slate.
        if self.filename is not None and force_new and os.path.exists(self.filename):
            os.remove(self.filename)

        if self.filename is not None and os.path.exists(self.filename) and not force_new:
            # === LOAD FULL INDEX INTO MEMORY (retrieval path) ===
            print("index already exists, loading...")
            self._load_full_index(dim_voc)
            print("done loading index...")
        else:
            # === INITIALIZE NEW / STREAMING INDEX (indexing path) ===
            print("initializing new index...")
            self.index_doc_id = defaultdict(lambda: array.array("I"))
            self.index_doc_value = defaultdict(lambda: array.array("f"))
            # Create empty file with metadata if needed; datasets are created
            # lazily on first flush to keep startup fast.
            if self.filename is not None and not os.path.exists(self.filename):
                with h5py.File(self.filename, "w") as f:
                    pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_full_index(self, dim_voc):
        """Load entire index from disk into numpy arrays (retrieval path)."""
        self.index_doc_id = dict()
        self.index_doc_value = dict()
        with h5py.File(self.filename, "r") as f:
            if dim_voc is not None:
                dim = dim_voc
            else:
                dim = f["dim"][()]
            for key in tqdm(range(dim)):
                name_id = f"index_doc_id_{key}"
                name_val = f"index_doc_value_{key}"
                if name_id in f and name_val in f:
                    self.index_doc_id[key] = np.array(f[name_id], dtype=np.int32)
                    self.index_doc_value[key] = np.array(f[name_val], dtype=np.float32)
                else:
                    self.index_doc_id[key] = np.array([], dtype=np.int32)
                    self.index_doc_value[key] = np.array([], dtype=np.float32)
        # ``nb_docs`` is only needed for retrieval; it will be set by the
        # caller based on doc_ids length when loading.

    def _flush_buffers(self):
        """Append current in-memory posting buffers to HDF5 and clear them.

        This is only used during indexing (when we are in streaming mode).
        It keeps memory bounded: after each flush, only new postings
        accumulated since the previous flush remain in memory.
        """
        if self.filename is None or len(self.index_doc_id) == 0:
            return

        with h5py.File(self.filename, "a") as f:
            # Determine dim = max posting-list id + 1 across all datasets.
            all_keys = list(self.index_doc_id.keys())
            if "dim" in f:
                current_dim = int(f["dim"][()])
            else:
                current_dim = 0
                f.create_dataset("dim", data=0)

            for key in all_keys:
                doc_ids_buf = self.index_doc_id[key]
                values_buf = self.index_doc_value[key]
                if len(doc_ids_buf) == 0:
                    continue

                doc_ids_np = np.frombuffer(doc_ids_buf, dtype=np.uint32).astype(np.int32)
                values_np = np.frombuffer(values_buf, dtype=np.float32)

                name_id = f"index_doc_id_{key}"
                name_val = f"index_doc_value_{key}"

                if name_id in f and name_val in f:
                    # Extend existing datasets along first axis.
                    old_len = f[name_id].shape[0]
                    new_len = old_len + doc_ids_np.shape[0]
                    f[name_id].resize((new_len,))
                    f[name_val].resize((new_len,))
                    f[name_id][old_len:new_len] = doc_ids_np
                    f[name_val][old_len:new_len] = values_np
                else:
                    # Create new datasets with chunking and maxshape=None so
                    # we can extend them later.
                    maxshape = (None,)
                    f.create_dataset(name_id, data=doc_ids_np, maxshape=maxshape, chunks=True)
                    f.create_dataset(name_val, data=values_np, maxshape=maxshape, chunks=True)

                if key + 1 > current_dim:
                    current_dim = key + 1

            # Update dim.
            f["dim"][...] = int(current_dim)

        # Clear in-memory buffers to free RAM.
        self.index_doc_id = defaultdict(lambda: array.array("I"))
        self.index_doc_value = defaultdict(lambda: array.array("f"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_batch_document(self, row, col, data, n_docs=-1):
        """Add a batch of documents to the index.

        When in streaming mode (indexing path), postings are accumulated
        in small in-memory buffers and periodically flushed to disk via
        ``save()``. When loaded for retrieval, postings are already numpy
        arrays and ``add_batch_document`` is not used.
        """
        if n_docs < 0:
            self.n += len(set(row))
        else:
            self.n += n_docs

        # Accumulate in memory (streaming mode).
        for doc_id, dim_id, value in zip(row, col, data):
            self.index_doc_id[int(dim_id)].append(int(doc_id))
            self.index_doc_value[int(dim_id)].append(float(value))

        # If buffers grow too large, caller should invoke save(), which will
        # trigger ``_flush_buffers``.

    def __len__(self):
        return len(self.index_doc_id)

    def nb_docs(self):
        return self.n

    def save(self, dim=None):
        """Flush current buffers to disk and update index metadata.

        For streaming indexing, this is called periodically; for a final
        save at the end of indexing; and is a no-op for fully loaded
        retrieval indexes.
        """
        # Only streaming indexes (initialized with force_new=True) maintain
        # array.array buffers; if we are in retrieval mode, nothing to do.
        if not isinstance(self.index_doc_id, defaultdict):
            return

        # Flush current buffers to HDF5 and clear them.
        self._flush_buffers()

        if self.index_path is None:
            return

        # Re-open file in read mode to compute index distribution.
        index_dist = {}
        with h5py.File(self.filename, "r") as f:
            if dim is not None:
                dim_val = int(dim)
            else:
                dim_val = int(f["dim"][()]) if "dim" in f else 0
            for key in range(dim_val):
                name_id = f"index_doc_id_{key}"
                if name_id in f:
                    index_dist[int(key)] = int(f[name_id].shape[0])
                else:
                    index_dist[int(key)] = 0

        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))
