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
    """
    Inverted index implementation using arrays for efficient sparse retrieval.
    
    This class maintains an inverted index where:
    - Each vocabulary dimension (token) maps to lists of document IDs and their corresponding values
    - Uses HDF5 for persistent storage and numpy arrays for efficient computation
    - Supports both creation of new indices and loading existing ones
    
    The index structure:
    - index_doc_id[dim_id] -> array of document IDs containing this dimension
    - index_doc_value[dim_id] -> array of values for this dimension in corresponding documents
    
    Attributes:
        index_path (str): Directory path for storing index files
        filename (str): Full path to the HDF5 index file
        index_doc_id (dict): Maps dimension IDs to arrays of document IDs
        index_doc_value (dict): Maps dimension IDs to arrays of corresponding values
        n (int): Total number of documents in the index
    """

    def __init__(self, index_path=None, force_new=False, filename="array_index.h5py", dim_voc=None):
        """
        Initialize the inverted index.
        
        Args:
            index_path (str, optional): Directory to store/load index files. If None, creates in-memory index.
            force_new (bool): If True, creates new index even if existing one found. Defaults to False.
            filename (str): Name of the HDF5 file for persistent storage. Defaults to "array_index.h5py".
            dim_voc (int, optional): Vocabulary dimension size. If None, reads from existing index.
        """
        if index_path is not None:
            self.index_path = index_path
            if not os.path.exists(index_path):
                os.makedirs(index_path)
            self.filename = os.path.join(self.index_path, filename)
            if os.path.exists(self.filename) and not force_new:
                print("index already exists, loading...")
                self.file = h5py.File(self.filename, "r")
                if dim_voc is not None:
                    dim = dim_voc
                else:
                    dim = self.file["dim"][()]
                self.index_doc_id = dict()
                self.index_doc_value = dict()
                for key in tqdm(range(dim)):
                    try:
                        self.index_doc_id[key] = np.array(self.file["index_doc_id_{}".format(key)],
                                                          dtype=np.int32)
                        # ideally we would not convert to np.array() but we cannot give pool an object with hdf5
                        self.index_doc_value[key] = np.array(self.file["index_doc_value_{}".format(key)],
                                                             dtype=np.float32)
                    except:
                        self.index_doc_id[key] = np.array([], dtype=np.int32)
                        self.index_doc_value[key] = np.array([], dtype=np.float32)
                self.file.close()
                del self.file
                print("done loading index...")
                doc_ids = pickle.load(open(os.path.join(self.index_path, "doc_ids.pkl"), "rb"))
                self.n = len(doc_ids)
            else:
                self.n = 0
                print("initializing new index...")
                self.index_doc_id = defaultdict(lambda: array.array("I"))
                self.index_doc_value = defaultdict(lambda: array.array("f"))
        else:
            self.n = 0
            print("initializing new index...")
            self.index_doc_id = defaultdict(lambda: array.array("I"))
            self.index_doc_value = defaultdict(lambda: array.array("f"))

    def add_batch_document(self, row, col, data, n_docs=-1):
        """
        Add a batch of documents to the index in sparse format.
        
        Args:
            row (array-like): Document IDs for each non-zero entry
            col (array-like): Dimension IDs (vocabulary tokens) for each non-zero entry  
            data (array-like): Values for each non-zero entry
            n_docs (int): Number of unique documents being added. If -1, computed from unique row values.
        """
        if n_docs < 0:
            self.n += len(set(row))
        else:
            self.n += n_docs
        for doc_id, dim_id, value in zip(row, col, data):
            self.index_doc_id[dim_id].append(doc_id)
            self.index_doc_value[dim_id].append(value)

    def __len__(self):
        """Return the number of vocabulary dimensions in the index."""
        return len(self.index_doc_id)

    def nb_docs(self):
        """Return the total number of documents in the index."""
        return self.n

    def save(self, dim=None):
        """
        Save the index to disk in HDF5 format.
        
        Converts internal array.array structures to numpy arrays and saves to HDF5.
        Also saves index distribution statistics as JSON.
        
        Args:
            dim (int, optional): Vocabulary dimension size to save. If None, uses current index size.
        """
        print("converting to numpy")
        for key in tqdm(list(self.index_doc_id.keys())):
            self.index_doc_id[key] = np.array(self.index_doc_id[key], dtype=np.int32)
            self.index_doc_value[key] = np.array(self.index_doc_value[key], dtype=np.float32)
        print("save to disk")
        with h5py.File(self.filename, "w") as f:
            if dim:
                f.create_dataset("dim", data=int(dim))
            else:
                f.create_dataset("dim", data=len(self.index_doc_id.keys()))
            for key in tqdm(self.index_doc_id.keys()):
                f.create_dataset("index_doc_id_{}".format(key), data=self.index_doc_id[key])
                f.create_dataset("index_doc_value_{}".format(key), data=self.index_doc_value[key])
            f.close()
        print("saving index distribution...")  # => size of each posting list in a dict
        index_dist = {}
        for k, v in self.index_doc_id.items():
            index_dist[int(k)] = len(v)
        json.dump(index_dist, open(os.path.join(self.index_path, "index_dist.json"), "w"))
