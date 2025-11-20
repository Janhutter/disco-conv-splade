from github import Github
import os, base64
import json

token = os.environ.get("GITHUB_TOKEN", None)
g = Github(token) if token else Github()
repo = g.get_repo("irlabamsterdam/iKAT")
branch = "main"
path = "2023/data"

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

# download_dir(repo, path, "DATA/ikat23")

# need to have access to https://ikattrecweb.grill.science/UvA/

# do till file 15

import subprocess

def download_file(url, out_path, username, password):
    cmd = [
        "wget",
        f"--user={username}",
        f"--password={password}",
        "-O", out_path,
        url
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error downloading:", url)
        print(result.stderr)
    else:
        print("Downloaded:", url)


def download():
    USERNAME = os.environ["IKAT_USERNAME"]
    PASSWORD = os.environ["IKAT_PASSWORD"]
    AUTH = f"--user={USERNAME} --password={PASSWORD}"

    os.makedirs("DATA/ikat23", exist_ok=True)

    for i in range(16):  # 00–15 inclusive
        idx = f"{i:02d}"
        bz_path = f"DATA/ikat23/ikat23_2023_passages_{idx}.jsonl.bz2"
        url = f"https://ikattrecweb.grill.science/UvA/indices/ikat_2023_passages_{idx}.jsonl.bz2"

        download_file(url, bz_path, USERNAME, PASSWORD)
        subprocess.run(["bzip2", "-f", "-d", bz_path])
download()

with open("DATA/ikat23/2023_test_topics_psg_text.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

docs_text = {}
new_ids = {}
new_ids_no_passage = {}
new_id = 0
for line in lines:
    entry = json.loads(line)
    doc_id = entry["doc_id"]
    passage = entry["passage_id"]
    doc_id = f"{doc_id}:{passage}"
    text = entry["passage_text"]
    docs_text[new_id] = text
    new_ids[doc_id] = new_id
    new_ids_no_passage[doc_id.split(":")[0]] = new_id
    new_id += 1

# write docs_text to tsv
with open("DATA/ikat23/ikat23_test_passages.tsv", "w", encoding="utf-8") as f:
    for doc_id in docs_text:
        text = docs_text[doc_id].replace("\n", " ").replace("\t", " ")
        f.write(f"{doc_id}\t{text}\n")

with open("DATA/ikat23/2023_train_topics_psg_text.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
docs_text_train = {}
for line in lines:
    entry = json.loads(line)
    doc_id = entry["doc_id"]
    text = entry["passage_text"]
    docs_text_train[new_id] = text
    new_ids[doc_id] = new_id
    new_ids_no_passage[doc_id.split(":")[0]] = new_id
    new_id += 1

with open("DATA/ikat23/ikat23_train_passages.tsv", "w", encoding="utf-8") as f:
    for doc_id in docs_text_train:
        text = docs_text_train[doc_id].replace("\n", " ").replace("\t", " ")
        f.write(f"{doc_id}\t{text}\n")


with open("DATA/ikat23/2023_test_topics.json", 'r', encoding="utf-8") as f:
    data = json.load(f)

turns = 0
with open("DATA/ikat23/queries_rowid_dev_all.tsv", "w", encoding="utf-8") as f_out_test, open("DATA/ikat23/qrel_rowid_dev.json", "w", encoding="utf-8") as qrel_test:
    for conversation in data:
        older_turns = []
        for turn in conversation['turns']:
            query = turn['utterance']
            docs = turn['response_provenance']
            response = turn['response']
            human_response = turn['resolved_utterance']
            older_turns.append(query)
            new_doc_ids = [new_ids[doc] for doc in docs if doc in new_ids]
            # first entry is newest turn
            conv = ' [SEP] '.join(older_turns[::-1])
            f_out_test.write(f"{turns}\t{conv}\n")

            older_turns.append(response)

            qrel_entry = {
                new_doc_id: 1 for new_doc_id in new_doc_ids
            }
            qrel_test.write(json.dumps({str(turns): qrel_entry}) + "\n")
            turns += 1

with open("DATA/ikat23/queries_rowid_train_all.tsv", "w", encoding="utf-8") as f_out_train, open("DATA/ikat23/qrel_rowid_train.json", "w", encoding="utf-8") as qrel_train:
    qrel_train.write("{\n")  # create empty file if not exists
    for conversation in data:
        older_turns = []
        for turn in conversation['turns']:
            query = turn['utterance']
            docs = turn['response_provenance']
            new_doc_ids = [new_ids[doc] for doc in docs if doc in new_ids]
            if new_doc_ids == []:
                continue
            response = turn['response']
            human_response = turn['resolved_utterance']
            older_turns.append(query)
                
            # first entry is newest turn
            conv = ' [SEP] '.join(older_turns[::-1])
            f_out_train.write(f"{turns}\t{conv}\n")

            older_turns.append(response)

            qrel_entry = {
                new_doc_id: 1 for new_doc_id in new_doc_ids
            }
            qrel_train.write("\"" + str(turns) + "\": " + json.dumps(qrel_entry) + ",\n")
            turns += 1
    qrel_train.write("}\n")