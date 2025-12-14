from github import Github
import os, base64
import json

token = os.environ.get("GITHUB_TOKEN", None)
g = Github(token) if token else Github()
repo = g.get_repo("irlabamsterdam/iKAT")
branch = "main"
path_2023 = "2023/data"
path_2024 = "2024/data"

base_folder = "/scratch-shared/disco/"

def download_dir(repo, path, local_dir, ref=branch):
    contents = repo.get_contents(path, ref=ref)
    for content in contents:
        if content.type == "dir":
            new_local = os.path.join(local_dir, content.name)
            os.makedirs(new_local, exist_ok=True)
            download_dir(repo, content.path, new_local, ref)
        else:
            print("Downloading:", content.path)
            filedata = base64.b64decode(content.content)
            os.makedirs(local_dir, exist_ok=True)
            with open(os.path.join(local_dir, content.name), "wb") as f:
                f.write(filedata)

# Example (unused in current pipeline):
# download_dir(repo, path_2023, "DATA/ikat23")

# need to have access to https://ikattrecweb.grill.science/UvA/

# do till file 15

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_file(url, out_path, username, password):
    """Download a single file with wget if the target .bz2 and .jsonl don't exist."""
    # If decompressed file already exists, skip download
    if os.path.exists(out_path[:-4]):
        print("Skip, already decompressed:", out_path[:-4])
        return

    # If .bz2 already exists, we don't re-download, just keep it for decompression
    if os.path.exists(out_path):
        print("Skip download, .bz2 already present:", out_path)
        return

    cmd = [
        "wget",
        f"--user={username}",
        f"--password={password}",
        "-O",
        out_path,
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error downloading:", url)
        print(result.stderr)
    else:
        print("Downloaded:", url)


def decompress_and_cleanup(bz_path):
    """Decompress a .bz2 file (if needed) and remove the archive to save space."""
    jsonl_path = bz_path[:-4]

    # If decompressed exists, just remove leftover .bz2 if present
    if os.path.exists(jsonl_path):
        if os.path.exists(bz_path):
            os.remove(bz_path)
        print("Already decompressed:", jsonl_path)
        return

    if not os.path.exists(bz_path):
        print("Missing .bz2, cannot decompress:", bz_path)
        return

    # -f to overwrite any existing partial file, -d to decompress
    result = subprocess.run(["bzip2", "-f", "-d", bz_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error decompressing:", bz_path)
        print(result.stderr)
    else:
        print("Decompressed:", jsonl_path)
        # bzip2 -d removes the .bz2 by default; just in case, clean up
        if os.path.exists(bz_path):
            os.remove(bz_path)


def download_ikat23_collection():
    USERNAME = os.environ["IKAT_USERNAME"]
    PASSWORD = os.environ["IKAT_PASSWORD"]

    os.makedirs(f"{base_folder}DATA/ikat23", exist_ok=True)

    # Prepare download tasks for 2023 collection
    tasks = []
    for i in range(16):  # 00–15 inclusive
        idx = f"{i:02d}"
        bz_path = f"{base_folder}DATA/ikat23/ikat23_2023_passages_{idx}.jsonl.bz2"
        url = f"https://ikattrecweb.grill.science/UvA/indices/ikat_2023_passages_{idx}.jsonl.bz2"
        tasks.append((url, bz_path))

    # Parallel download using a thread pool
    max_workers = min(8, len(tasks))  # avoid spawning too many threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_file, url, bz_path, USERNAME, PASSWORD)
            for url, bz_path in tasks
        ]
        for future in as_completed(futures):
            # Trigger any exceptions raised during download
            _ = future.result()
    # Decompress sequentially (CPU/disk-bound, but lightweight vs network)
    for _, bz_path in tasks:
        decompress_and_cleanup(bz_path)



def process_ikat23():
    """End-to-end processing for iKAT23 (2023 topics)."""

    # Combine the jsonl files from the collection into a single TSV file
    # with columns: id \t text. We stream line-by-line to avoid loading
    # large files into memory.
    passages_tsv = f"{base_folder}DATA/ikat23/passages.tsv"

    # if not os.path.exists(passages_tsv):
        # os.makedirs(f"{base_folder}DATA/ikat23", exist_ok=True)
    with open(passages_tsv, "w", encoding="utf-8") as out_f:
        for i in range(16):  # shards 00–15
            idx = f"{i:02d}"
            shard_path = f"{base_folder}DATA/ikat23/ikat23_2023_passages_{idx}.jsonl"
            if not os.path.exists(shard_path):
                print("Missing shard, skipping:", shard_path)
                continue

            print("Processing shard:", shard_path)
            with open(shard_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        print("JSON decode error in", shard_path)
                        continue

                    doc_id = obj.get("id")
                    contents = obj.get("contents")
                    if doc_id is None or contents is None:
                        continue

                    contents_str = str(contents).replace("\n", " ").strip()
                    out_f.write(f"{doc_id}\t{contents_str}\n")
    # else:
    #     print("passages.tsv already exists, skipping collection merge.")

    doc_ids = []
    with open(passages_tsv, "r", encoding="utf-8") as f:
        for line in f:
            doc_id, _ = line.split("\t", 1)
            doc_ids.append(doc_id)
    import pickle
    with open(f"/scratch-shared/disco/DATA/ikat23/splade_index/splade_index/doc_ids.pkl", "wb") as f:
        pickle.dump(doc_ids, f)

    # Dev topics (test split)
    with open(f"{base_folder}DATA/ikat23/2023_test_topics.json", 'r', encoding="utf-8") as f:
        data = json.load(f)

    turns = 0
    qrel_dev = {}
    with open(f"{base_folder}DATA/ikat23/queries_rowid_dev_all.tsv", "w", encoding="utf-8") as f_out_test, \
         open(f"{base_folder}DATA/ikat23/human_rewrites_dev.tsv", "w", encoding="utf-8") as f_out_dev_human:
        for conversation in data:
            older_turns = []
            for turn in conversation['turns']:
                query = turn['utterance']
                docs = turn['response_provenance']
                response = turn['response']
                human_response = turn['resolved_utterance']
                older_turns.append(query)
                # response_provenance uses doc:passage ids; map directly
                # Skip turns without any annotated relevance labels
                if not docs:
                    continue

                # check if any of the docs are missing from doc_ids
                new_docs = []
                for doc in docs:
                    if doc in doc_ids:
                        new_docs.append(doc)
                docs = new_docs

                if not docs:
                    continue

                # first entry is newest turn
                conv = ' [SEP] '.join(older_turns[::-1])
                f_out_test.write(f"{turns}\t{conv}\n")
                # aligned human rewrite for this row id
                if human_response is not None:
                    human_rewrite = str(human_response).replace("\n", " ").strip()
                else:
                    human_rewrite = ""
                f_out_dev_human.write(f"{turns}\t{human_rewrite}\n")

                older_turns.append(response)

                qrel_entry = {
                    new_doc_id: 1 for new_doc_id in docs
                }
                qrel_dev[str(turns)] = qrel_entry
                turns += 1

    # write single-line JSON object for dev qrels
    with open(f"{base_folder}DATA/ikat23/qrel_rowid_dev.json", "w", encoding="utf-8") as qrel_test:
        qrel_test.write(json.dumps(qrel_dev))

    # Train topics (separate file to avoid reusing dev split)
    with open(f"{base_folder}DATA/ikat23/2023_train_topics.json", 'r', encoding="utf-8") as f:
        data = json.load(f)

    qrel_train_obj = {}
    with open(f"{base_folder}DATA/ikat23/queries_rowid_train_all.tsv", "w", encoding="utf-8") as f_out_train, \
         open(f"{base_folder}DATA/ikat23/human_rewrites_train.tsv", "w", encoding="utf-8") as f_out_train_human:
        for conversation in data:
            older_turns = []
            for turn in conversation['turns']:
                query = turn['utterance']
                docs = turn['response_provenance']
                # Train also uses doc:passage ids according to the
                # official format; map with the same dictionary.
                # new_doc_ids = [new_ids[doc] for doc in docs if doc in new_ids]
                # Skip turns without any annotated relevance labels
                if not docs:
                    continue


                new_docs = []
                for doc in docs:
                    if doc in doc_ids:
                        new_docs.append(doc)
                docs = new_docs

                if not docs:
                    continue

                response = turn['response']
                human_response = turn['resolved_utterance']
                older_turns.append(query)

                # first entry is newest turn
                conv = ' [SEP] '.join(older_turns[::-1])
                f_out_train.write(f"{turns}\t{conv}\n")

                if human_response is not None:
                    human_rewrite = str(human_response).replace("\n", " ").strip()
                else:
                    human_rewrite = ""
                f_out_train_human.write(f"{turns}\t{human_rewrite}\n")

                older_turns.append(response)

                qrel_entry = {
                    new_doc_id: 1 for new_doc_id in docs
                }
                qrel_train_obj[str(turns)] = qrel_entry
                turns += 1

    # write single-line JSON object for train qrels
    with open(f"{base_folder}DATA/ikat23/qrel_rowid_train.json", "w", encoding="utf-8") as qrel_train:
        qrel_train.write(json.dumps(qrel_train_obj))


def process_ikat24():
    """Processing for iKAT24, reusing the iKAT23 passage collection.

    Assumes 2024 topics live under DATA/ikat24 but share the same
    passage IDs/contents as 2023.
    """
    # new_ids, new_ids_no_passage = build_id_mapping_ikat23()

    with open(f"{base_folder}DATA/ikat24/2024_test_topics.json", 'r', encoding="utf-8") as f:
        data_2024 = json.load(f)

    doc_ids = []
    with open(f"{base_folder}DATA/ikat23/passages.tsv", "r", encoding="utf-8") as f:
        for line in f:
            doc_id, _ = line.split("\t", 1)
            doc_ids.append(doc_id)

    turns = 0
    qrel_dev_2024 = {}
    with open(f"{base_folder}DATA/ikat24/queries_rowid_dev_all.tsv", "w", encoding="utf-8") as f_out_test, \
         open(f"{base_folder}DATA/ikat24/human_rewrites_dev.tsv", "w", encoding="utf-8") as f_out_dev_human:
        for conversation in data_2024:
            older_turns = []
            for turn in conversation['turns']:
                query = turn['utterance']
                docs = turn['response_provenance']
                response = turn['response']
                human_response = turn['resolved_utterance']
                older_turns.append(query)
                # 2024 response_provenance is also in doc:passage form

                # Skip turns without any annotated relevance labels
                if not docs:
                    continue

                new_docs = []
                for doc in docs:
                    if doc in doc_ids:
                        new_docs.append(doc)
                docs = new_docs

                if not docs:
                    continue

                conv = ' [SEP] '.join(older_turns[::-1])
                f_out_test.write(f"{turns}\t{conv}\n")

                if human_response is not None:
                    human_rewrite = str(human_response).replace("\n", " ").strip()
                else:
                    human_rewrite = ""
                f_out_dev_human.write(f"{turns}\t{human_rewrite}\n")

                older_turns.append(response)

                qrel_entry = {
                    new_doc_id: 1 for new_doc_id in docs
                }
                qrel_dev_2024[str(turns)] = qrel_entry
                turns += 1

    # write single-line JSON object for 2024 dev qrels
    with open(f"{base_folder}DATA/ikat24/qrel_rowid_dev.json", "w", encoding="utf-8") as qrel_test:
        qrel_test.write(json.dumps(qrel_dev_2024))

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_splade_index():
    urls = [
        "https://ikattrecweb.grill.science/UvA/indices/splade_index.tar.bz2.partaa",
        "https://ikattrecweb.grill.science/UvA/indices/splade_index.tar.bz2.partab",
        "https://ikattrecweb.grill.science/UvA/indices/splade_index.tar.bz2.partac",
        "https://ikattrecweb.grill.science/UvA/indices/splade_index.tar.bz2.partad",
        "https://ikattrecweb.grill.science/UvA/indices/splade_index.tar.bz2.partae",
        "https://ikattrecweb.grill.science/UvA/indices/splade_index.tar.bz2.partaf",
        "https://ikattrecweb.grill.science/UvA/indices/splade_index.tar.bz2.partag",
        "https://ikattrecweb.grill.science/UvA/indices/splade_index.tar.bz2.partah",
        "https://ikattrecweb.grill.science/UvA/indices/splade_index.tar.bz2.partai",
    ]

    USERNAME = os.environ["IKAT_USERNAME"]
    PASSWORD = os.environ["IKAT_PASSWORD"]

    out_dir = f"{base_folder}DATA/ikat23/splade_index"
    os.makedirs(out_dir, exist_ok=True)

    def download_one(url):
        part_name = url.split("/")[-1]
        part_path = f"{out_dir}/{part_name}"
        download_file(url, part_path, USERNAME, PASSWORD)
        return part_path

    # part_paths = []
    # Use up to 8 threads—tune as needed
    # with ThreadPoolExecutor(max_workers=8) as ex:
    #     futures = {ex.submit(download_one, url): url for url in urls}
    #     for fut in as_completed(futures):
    #         part_paths.append(fut.result())



    # Sort parts to ensure correct order (ThreadPool finishes unordered)
    # part_paths.sort()

    # Combine parts
    combined_path = f"{out_dir}/splade_index.tar.bz2"
    # with open(combined_path, "wb") as outfile:
    #     for part_path in part_paths:
    #         with open(part_path, "rb") as infile:
    #             outfile.write(infile.read())

    # Extract
    # subprocess.run(
    #     ["tar", "-xvjf", combined_path, "-C", out_dir],
    #     check=True
    # )
    new_out = f"{base_folder}DATA/ikat23/splade_index_new"
    subprocess.run(
        ["tar", "--use-compress-program=lbzip2", "-xvf", combined_path, "-C", new_out],
        check=True
    )

    # Optionally clean up
    # for p in part_paths:
    #     os.remove(p)
    # os.remove(combined_path)

    # write a pkl file with every row containing the doc ids


if __name__ == "__main__":
    process_ikat23()
    process_ikat24()
    # download_splade_index()