import glob
import json
import os
import pickle

import numpy as np

from .indexing.inverted_index import IndexDictOfArray


def _discover_shard_dirs(index_dir):
    pattern = os.path.join(index_dir, "rank*")
    shard_dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    shard_dirs.sort()
    return shard_dirs


def merge_index_shards(index_dir: str):
    """Merge per-rank index shards into a single index.

    Expects the following per-rank files produced by ``SparseIndexing``:

    - ``index_dir/rank{r}/array_index.h5py``
    - ``index_dir/doc_ids_rank{r}.pkl``
    - optionally ``index_dir/index_stats_rank{r}.json``

    Produces in ``index_dir`` (without any rank suffix):

    - ``array_index.h5py``
    - ``doc_ids.pkl``
    - optionally ``index_stats.json`` (if all shard stats are present)
    """

    index_dir = os.path.abspath(index_dir)
    shard_dirs = _discover_shard_dirs(index_dir)
    if not shard_dirs:
        raise ValueError(f"No shard directories found under {index_dir} (expected rank*/)")

    print(f"Merging {len(shard_dirs)} shards from {index_dir}...")

    # Load all per-rank doc id lists in rank order and compute offsets.
    shard_doc_ids = []
    shard_sizes = []
    for shard_dir in shard_dirs:
        # infer rank from directory name, but only for logging; ordering already sorted
        base = os.path.basename(shard_dir)
        rank_suffix = base.replace("rank", "")
        doc_ids_path = os.path.join(index_dir, f"doc_ids_rank{rank_suffix}.pkl")
        if not os.path.exists(doc_ids_path):
            raise FileNotFoundError(f"Missing doc ids file: {doc_ids_path}")
        ids = pickle.load(open(doc_ids_path, "rb"))
        shard_doc_ids.append(ids)
        shard_sizes.append(len(ids))
        print(f"  shard {base}: {len(ids)} documents")

    offsets = np.cumsum([0] + shard_sizes[:-1])

    # Initialize merged index living directly in index_dir (no rank suffix).
    merged_index = IndexDictOfArray(index_path=index_dir, force_new=True)

    # For each shard, load its index and add with shifted doc ids.
    from .indexing.inverted_index import IndexDictOfArray as ShardIndex

    for shard_dir, offset in zip(shard_dirs, offsets):
        print(f"Loading shard index from {shard_dir} with offset {offset}...")
        shard_index = ShardIndex(index_path=shard_dir, force_new=False)
        # For each posting list (dimension), append shifted ids and values.
        for dim_id, doc_ids_arr in shard_index.index_doc_id.items():
            if len(doc_ids_arr) == 0:
                continue
            shifted_ids = doc_ids_arr.astype(np.int64) + int(offset)
            merged_index.add_batch_document(
                row=shifted_ids.astype(np.int32),
                col=np.full_like(shifted_ids, dim_id, dtype=np.int32),
                data=shard_index.index_doc_value[dim_id].astype(np.float32),
                n_docs=-1,
            )

    # Save merged index to disk without rank suffix.
    merged_index.save()

    # Merge doc ids in shard order and save without rank suffix.
    all_doc_ids = []
    for ids in shard_doc_ids:
        all_doc_ids.extend(ids)
    doc_ids_out = os.path.join(index_dir, "doc_ids.pkl")
    pickle.dump(all_doc_ids, open(doc_ids_out, "wb"))
    print(f"Saved merged doc ids to {doc_ids_out} ({len(all_doc_ids)} documents)")

    # Optionally merge stats if present for all shards.
    shard_stats = []
    for shard_dir in shard_dirs:
        base = os.path.basename(shard_dir)
        rank_suffix = base.replace("rank", "")
        stats_path = os.path.join(index_dir, f"index_stats_rank{rank_suffix}.json")
        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                shard_stats.append(json.load(f))
        else:
            shard_stats = []
            break

    if shard_stats:
        merged_stats = {}
        for s in shard_stats:
            for k, v in s.items():
                merged_stats[k] = merged_stats.get(k, 0.0) + float(v)
        # average over shards (same behavior as averaging over loaders per rank)
        for k in merged_stats:
            merged_stats[k] /= float(len(shard_stats))
        stats_out = os.path.join(index_dir, "index_stats.json")
        with open(stats_out, "w") as f:
            json.dump(merged_stats, f)
        print(f"Saved merged stats to {stats_out}")

    print("Done merging shards.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Merge per-rank sparse index shards into a single index.")
    parser.add_argument("index_dir", type=str, help="Directory containing rank* subdirectories and *_rank*.pkl/json files")
    args = parser.parse_args()

    merge_index_shards(args.index_dir)
