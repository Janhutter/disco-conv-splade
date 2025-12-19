from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi
import pandas as pd
import os
import json


def detect_format(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".json", ".jsonl"]:
        return "json"
    if ext == ".tsv":
        return ("csv", "\t")
    if ext == ".csv":
        return ("csv", ",")
    if ext == ".parquet":
        return "parquet"
    raise ValueError(f"Unsupported file type: {ext}")


def load_tsv_or_csv(path, delimiter):
    """
    Safely load CSV/TSV files with automatic header detection and row-repair.
    Returns a HuggingFace Dataset.
    """

    # First read a few lines to detect header
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().split(delimiter)

    # Detect header: if every field is non-numeric and contains letters → assume header
    header_is_present = all(any(c.isalpha() for c in col) for col in first_line)

    if not header_is_present:
        print(f"[Info] No header detected in {path}. Assigning default header ['id','contents']")
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            names=["id", "contents"],
            on_bad_lines="skip",
            engine="python",
        )
    else:
        print(f"[Info] Header detected in {path}: {first_line}")
        df = pd.read_csv(
            path,
            delimiter=delimiter,
            on_bad_lines="skip",
            engine="python",
        )

    from datasets import Dataset
    return Dataset.from_pandas(df, preserve_index=False)


def load_json(path):
    """
    Loads JSON/JSONL & automatically standardizes qrels:
    - 3 columns → ('query','doc','relevance')
    """
    file_size = os.path.getsize(path)

    # For very large JSON files we avoid the default Arrow JSON
    # reader, which can hit a 2GB per-column limit, by loading
    # line-by-line into smaller shards (expects JSONL-style input).
    if file_size > 2_000_000_000:  # ~2GB
        print(
            f"[Info] Large JSON detected ({file_size} bytes). "
            "Using shard-based JSONL loader to avoid ArrowCapacityError."
        )

        from datasets import Dataset, concatenate_datasets

        shards = []
        buffer = []
        shard_size = 100_000  # examples per shard

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"[Error] Failed to parse JSON line in {path}: {e}"
                    ) from e
                buffer.append(obj)
                if len(buffer) >= shard_size:
                    shards.append(Dataset.from_list(buffer))
                    buffer = []

        if buffer:
            shards.append(Dataset.from_list(buffer))

        if not shards:
            raise ValueError(f"No records found in large JSON file: {path}")

        if len(shards) == 1:
            ds = shards[0]
        else:
            ds = concatenate_datasets(shards)
    else:
        ds = load_dataset("json", data_files=path, split="train")

    # Auto-detect qrels-like structure
    sample = ds[0]

    if isinstance(sample, dict) and len(sample) == 3:
        keys = sorted(sample.keys())
        # e.g. ['doc','query','relevance']
        if set(keys) == {"query", "doc", "relevance"}:
            print(f"[Info] JSON appears to be qrels format (query, doc, relevance). OK.")
        else:
            print(f"[Info] JSON has 3 fields → Renaming to (query, doc, relevance)")
            new_columns = ["query", "doc", "relevance"]
            ds = ds.rename_columns({old: new for old, new in zip(sample.keys(), new_columns)})

    return ds


def load_single_file(path):
    fmt = detect_format(path)

    if fmt == "json":
        return load_json(path)

    if isinstance(fmt, tuple) and fmt[0] == "csv":
        _, delimiter = fmt
        return load_tsv_or_csv(path, delimiter)

    if fmt == "parquet":
        return load_dataset("parquet", data_files=path, split="train")

    raise ValueError("Unknown format.")


def upload_split(file_path, split, repo_id, private=False):
    """
    Upload a single file as a split.
    If the repo exists, merge with existing splits.
    """
    print(f"Loading file: {file_path}")
    new_split_dataset = load_single_file(file_path)

    try:
        existing = load_dataset(repo_id)
        print("Found existing dataset. Merging splits...")
        dataset_dict = DatasetDict(existing)
    except Exception:
        print("No existing dataset found. Creating new one...")
        dataset_dict = DatasetDict()

    dataset_dict[split] = new_split_dataset

    print(f"Pushing split '{split}' to {repo_id}")
    dataset_dict.push_to_hub(repo_id, private=private)
    print("Done!")



if __name__ == "__main__":
    # upload_split(
    #     file_path="/scratch-shared/disco/DATA/cast20/human_rewrites_dev.tsv",
    #     split="test",
    #     repo_id="JanHutter/cast20-human-rewrites-test",
    #     private=False,
    # )

    # upload_split(
    #     file_path="/scratch-shared/disco/DATA/cast20/passages.tsv",
    #     split="test",
    #     repo_id="JanHutter/cast20-collection",
    #     private=False,
    # )

    # upload_split(
    #     file_path="/scratch-shared/disco/DATA/cast20/queries_rowid_dev_all.tsv",
    #     split="test",
    #     repo_id="JanHutter/cast20-test-queries-all",
    #     private=False,
    # )

    # upload_split(
    #     file_path="/scratch-shared/disco/DATA/cast20/queries_rowid_dev_last.tsv",
    #     split="test",
    #     repo_id="JanHutter/cast20-test-queries-last",
    #     private=False,
    # )

    # upload_split(
    #     file_path="/scratch-shared/disco/DATA/cast20/t5_rewrites_dev.tsv",
    #     split="test",
    #     repo_id="JanHutter/cast20-t5-rewrites-test",
    #     private=False,
    # )

    # upload_split(
    #     file_path="/scratch-shared/disco/DATA/cast20/qrels_rowid_dev.json",
    #     split="test",
    #     repo_id="JanHutter/cast20-qrels",
    #     private=False,
    # )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_distil/distil_run_top_multi.json",
        split="train",
        repo_id="JanHutter/topiocqa-distil-run-top-multi-teacher",
        private=False,
    )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_distil/distil_run_top_multi.json",
        split="train",
        repo_id="JanHutter/topiocqa-distil-run-top-multi-teacher",
        private=False,
    )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_rewrites/rewrites_t5_test_last.json",
        split="test",
        repo_id="JanHutter/topiocqa-t5-rewrites-last",
        private=False,
    )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_rewrites/rewrites_t5_train_last.json",
        split="train",
        repo_id="JanHutter/topiocqa-t5-rewrites-last",
        private=False,
    )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_topics/qrel_rowid_dev.json",
        split="test",
        repo_id="JanHutter/topiocqa-qrels",
        private=False,
    )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_topics/qrel_rowid_train.json",
        split="train",
        repo_id="JanHutter/topiocqa-qrels",
        private=False,
    )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_topics/queries_rowid_dev_all.json",
        split="test",
        repo_id="JanHutter/topiocqa-queries-all",
        private=False,
    )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_topics/queries_rowid_train_all.json",
        split="train",
        repo_id="JanHutter/topiocqa-queries-all",
        private=False,
    )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_topics/queries_rowid_train_last.json",
        split="train",
        repo_id="JanHutter/topiocqa-queries-last",
        private=False,
    )

    upload_split(
        file_path="/scratch-shared/disco/DATA/topiocqa_topics/queries_rowid_dev_last.json",
        split="test",
        repo_id="JanHutter/topiocqa-queries-last",
        private=False,
    )

    # upload_split(
    #     file_path="/scratch-shared/disco/DATA/full_wiki_segments_topiocqa.tsv",
    #     split="train",
    #     repo_id="JanHutter/topiocqa-collection",
    #     private=False,
    # )

    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat23/human_rewrites_dev.tsv ',
        split='test',
        repo_id='JanHutter/ikat23-human-rewrites-test',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat23/human_rewrites_train.tsv ',
        split='train',
        repo_id='JanHutter/ikat23-human-rewrites-train',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat23/qrel_rowid_dev.json ',
        split='test',
        repo_id='JanHutter/ikat23-qrels',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat23/qrel_rowid_train.json ',
        split='train',
        repo_id='JanHutter/ikat23-qrels',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat23/queries_rowid_dev_all.json ',
        split='test',
        repo_id='JanHutter/ikat23-queries-all',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat23/queries_rowid_train_all.json ',
        split='train',
        repo_id='JanHutter/ikat23-queries-all',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat23_rewrites/rewrites_mistral_test_last.tsv',
        split='test',
        repo_id='JanHutter/ikat23-mistral-rewrites-last',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat23_rewrites/rewrites_mistral_train_last.tsv',
        split='train',
        repo_id='JanHutter/ikat23-mistral-rewrites-last',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat23_rewrites/rewrites_t5_test_last.tsv',
        split='test',
        repo_id='JanHutter/ikat23-t5-rewrites-last',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat24/human_rewrites_dev.tsv',
        split='test',
        repo_id='JanHutter/ikat24-human-rewrites-test',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat24/queryies_rowid_dev_all.json',
        split='test',
        repo_id='JanHutter/ikat24-queries-all',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat24/rewrites_t5_test_last.tsv',
        split='test',
        repo_id='JanHutter/ikat24-t5-rewrites-last',
        private=False,
    )
    upload_split(
        file_path='/scratch-shared/disco/DATA/ikat24/rewrites_mistral_test_last.tsv',
        split='test',
        repo_id='JanHutter/ikat24-mistral-rewrites-last',
        private=False,
    )
