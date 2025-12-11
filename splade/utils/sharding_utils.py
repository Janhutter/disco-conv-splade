# sharding_utils.py
import os
import json
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import h5py


def create_shards_sparse(doc_embeddings: sp.csr_matrix, n_shards: int, topk_dims: int = 100):
    """
    Assign documents to shards based on top-k nonzero dimensions with a progress bar.

    Returns:
        shard_ids (np.ndarray): Array of shape [n_docs], shard assignment
        shard_metadata (dict): For each shard: top dimensions (centroid approximation)
    """
    n_docs, dim_voc = doc_embeddings.shape
    shard_ids = np.zeros(n_docs, dtype=np.int32)
    shard_topdims = [set() for _ in range(n_shards)]

    for doc_idx in tqdm(range(n_docs), desc="Assigning docs to shards"):
        row = doc_embeddings.getrow(doc_idx)
        top_indices = row.indices[np.argsort(-row.data)[:topk_dims]]
        shard_idx = int(np.sum(top_indices)) % n_shards
        shard_ids[doc_idx] = shard_idx
        shard_topdims[shard_idx].update(top_indices)

    shard_metadata = {int(i): {"top_dims": sorted(list(shard_topdims[i]))} for i in range(n_shards)}
    return shard_ids, shard_metadata


def save_shard_metadata(shard_dir: str, shard_info: dict):
    os.makedirs(shard_dir, exist_ok=True)
    metadata_path = os.path.join(shard_dir, "shard_metadata.json")

    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    shard_info_clean = json.loads(json.dumps(shard_info, default=convert))
    with open(metadata_path, "w") as f:
        json.dump(shard_info_clean, f, indent=2)


def load_shard_metadata(shard_dir: str):
    metadata_path = os.path.join(shard_dir, "shard_metadata.json")
    with open(metadata_path, "r") as f:
        shard_info = json.load(f)
    return shard_info


def select_relevant_shards_sparse(query_embedding: sp.csr_matrix, shard_metadata: dict, top_m: int = None):
    """
    Select relevant shards based on overlap of query nonzero dims with shard top dims.
    """
    q_indices = set(query_embedding.indices)
    scores = []

    for shard_id, shard_info in shard_metadata.items():
        top_dims = set(shard_info["top_dims"])
        overlap = len(q_indices.intersection(top_dims))
        scores.append((shard_id, overlap))

    scores.sort(key=lambda x: x[1], reverse=True)
    if top_m is not None:
        scores = scores[:top_m]

    selected_shard_ids = [sid for sid, score in scores if score > 0]
    return selected_shard_ids


def save_sparse_shards_h5py_global(index, shard_ids, shard_dir_base):
    """
    Save shard-specific inverted index files, but keeping global doc IDs.
    
    Structure:
        shard_k.h5py:
            /index_doc_id/<dim>    -> global doc_ids belonging to shard k
            /index_doc_value/<dim> -> values aligned with doc_ids

    Also writes:
        shard_k_doclist.npy        -> global doc IDs for this shard
    
    Args:
        index: IndexDictOfArray-like object
        shard_ids: np.array of shape [n_docs], shard assignment for each doc
        shard_dir_base: directory to write shard files
    """

    n_docs = index.nb_docs()
    dim_voc = len(index.index_doc_id)

    os.makedirs(shard_dir_base, exist_ok=True)

    n_shards = shard_ids.max() + 1

    # Collect docs belonging to each shard (global ids)
    shard_docs = {i: [] for i in range(n_shards)}
    for doc_id in range(n_docs):
        shard_docs[shard_ids[doc_id]].append(doc_id)

    # Save each shard
    for shard in range(n_shards):
        docs = np.array(sorted(shard_docs[shard]), dtype=np.int32)

        shard_path = os.path.join(shard_dir_base, f"shard_{shard}.h5py")
        print(f"\nSaving shard {shard} ({len(docs)} docs) → {shard_path}")

        # save metadata doc list too
        np.save(os.path.join(shard_dir_base, f"shard_{shard}_doclist.npy"), docs)

        # Create HDF5 inverted index
        with h5py.File(shard_path, "w") as h5:
            grp_id = h5.create_group("index_doc_id")
            grp_val = h5.create_group("index_doc_value")

            # For each dimension, select only docs that belong to the shard
            for dim in tqdm(range(dim_voc), desc=f"Shard {shard}: dims"):
                doc_ids_global = index.index_doc_id[dim]
                values = index.index_doc_value[dim]

                # Mask: select entries with doc_id ∈ docs
                # Use a fast membership mask via binary search
                mask = np.in1d(doc_ids_global, docs, assume_unique=False)

                if not mask.any():
                    continue  # no docs from this dim fall into this shard

                doc_ids_shard = doc_ids_global[mask]
                vals_shard = values[mask]

                # Save under dimension name
                grp_id.create_dataset(
                    str(dim), data=doc_ids_shard, compression="gzip"
                )
                grp_val.create_dataset(
                    str(dim), data=vals_shard, compression="gzip"
                )

        print(f" → Shard {shard} saved successfully with {len(docs)} global doc IDs.")


# -----------------------------
# Standalone test
# -----------------------------# -----------------------------
# Main: Load array_index.h5py and run sharding
# -----------------------------
if __name__ == "__main__":
    from ..indexing.inverted_index import IndexDictOfArray
    import scipy.sparse as sp
    import numpy as np
    import os
    from tqdm.auto import tqdm

    # Paths
    index_dir = "DATA/topiocqa_subset/topiocqa_index_self"
    shard_base_dir = "./test_shards_sparse"

    # Step 1: Load the existing index
    print("Loading existing sparse index...")
    index = IndexDictOfArray(index_path=index_dir, filename="array_index.h5py")
    print(f"Loaded index with {index.nb_docs()} documents and {len(index)} dimensions.")

    # Step 2: Build sparse embeddings from the index
    n_docs = index.nb_docs()
    dim_voc = len(index.index_doc_id)
    doc_embeddings = sp.lil_matrix((n_docs, dim_voc), dtype=np.float32)

    print("Building sparse matrix from index...")
    for dim_id in tqdm(index.index_doc_id.keys(), desc="Populating sparse matrix"):
        doc_ids = index.index_doc_id[dim_id]
        values = index.index_doc_value[dim_id]
        doc_embeddings[doc_ids, dim_id] = values

    doc_embeddings = doc_embeddings.tocsr()
    print(f"Sparse document matrix shape: {doc_embeddings.shape}, nnz={doc_embeddings.nnz}")

    # Step 3: Create shards
    n_shards = 4
    topk_shard_dims = 100

    print("Creating shards...")
    shard_ids, shard_metadata = create_shards_sparse(doc_embeddings, n_shards=n_shards, topk_dims=topk_shard_dims)
    print(f"Created {n_shards} shards.")

    # Step 4: Save shard metadata
    for shard_idx in range(n_shards):
        shard_doc_mask = shard_ids == shard_idx
        shard_info = {
            "shard_id": int(shard_idx),
            "top_dims": shard_metadata[shard_idx]["top_dims"]
        }
        shard_dir = os.path.join(shard_base_dir, f"shard_{shard_idx}")
        save_shard_metadata(shard_dir, shard_info)
        print(f"Shard {shard_idx}: metadata saved with {np.sum(shard_doc_mask)} documents.")

    # Step 5: Test query routing
    query_density = 0.01
    print("\nGenerating random sparse query for routing test...")
    query_embedding = sp.random(1, dim_voc, density=query_density, format="csr", dtype=np.float32, random_state=99)

    # Load all shard metadata
    all_shard_metadata = {}
    for shard_idx in range(n_shards):
        shard_dir = os.path.join(shard_base_dir, f"shard_{shard_idx}")
        shard_info = load_shard_metadata(shard_dir)
        all_shard_metadata[shard_idx] = shard_info

    save_sparse_shards_h5py_global(index, shard_ids, shard_base_dir)

    selected_shards = select_relevant_shards_sparse(query_embedding, all_shard_metadata, top_m=2)
    print(f"Selected shards for the query: {selected_shards}")

    print("\nSparse sharding workflow completed successfully.")
