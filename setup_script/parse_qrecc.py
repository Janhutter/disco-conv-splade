import json
import os
import re

from datasets import load_dataset


dataset = load_dataset("slupart/qrecc")
base_folder = "/scratch-shared/disco/"

qrels_test = {}
qrels_train = {}

# Additional qrels keyed by conversation/turn id ("Conversation_no_Turn_no"),
# to mirror the reference snippet you provided.
qrels_test_turnid = {}
qrels_train_turnid = {}

queries_row_id_all_test = []
queries_row_id_all_train = []
queries_row_id_last_test = []
queries_row_id_last_train = []

# Queries keyed by Conversation_no_Turn_no, as requested. These mirror
# the text formatting of the row-id queries but use the turn id string
# instead of the numeric row id as key.
queries_turnid_all_test = []
queries_turnid_all_train = []
queries_turnid_last_test = []
queries_turnid_last_train = []


global row_id
row_id = 0



human_rewrites_test = []
human_rewrites_train = []


def trunc_content(context):
    truncated_context = []
    for i, c in enumerate(context):
        if i % 2 == 0:
            truncated_context.append(c[: (64 * 5)].strip())
        else:
            truncated_context.append(c[: (100 * 5)].strip())
    return truncated_context


import os
import json
import gzip

def iter_jsonl(path):
    """Yield JSON objects from .jsonl or .jsonl.gz files."""
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # skip malformed lines

def process_folder(folder):
    for fname in os.listdir(folder):
        if fname.endswith(".jsonl") or fname.endswith(".jsonl.gz"):
            yield from iter_jsonl(os.path.join(folder, fname))

output_path = "/scratch-shared/disco/DATA/qrecc/passages.tsv"

folders = [
    "/scratch-shared/disco/DATA/collection-paragraph/commoncrawl/",
    "/scratch-shared/disco/DATA/collection-paragraph/wayback/",
    "/scratch-shared/disco/DATA/collection-paragraph/wayback-backfill/",
]

with open(output_path, "w", encoding="utf-8") as out:
    for folder in folders:
        for obj in process_folder(folder):
            _id = obj.get("id", "")
            contents = obj.get("contents", "").replace("\t", " ").replace("\n", " ")
            out.write(f"{_id}\t{contents}\n")

# load passages dataset
passages_dataset = []
with open(output_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue  # skip empty lines
        parts = line.split("\t", 1)
        if len(parts) != 2:
            # optionally log or debug-print the bad line here
            continue  # skip malformed lines
        _id, contents = parts
        passages_dataset.append({"id": _id, "contents": contents})

# url2id = create_url2id_mapping(passages_dataset)


for split_name in ["train", "test"]:
    split = dataset[split_name]

    for entry in split:

        turn_no = entry["Turn_no"]
        # conversation/turn identifier as in the reference snippet
        turn_id = str(entry["Conversation_no"]) + "_" + str(entry["Turn_no"])

        context = list(entry["Context"])  # copy before reverse

        truth_passages = entry["Truth_passages"]
        truth_url = entry["Answer_URL"]

        # Skip queries with no annotated truth passages at all
        if truth_passages in (None, "", []):
            continue

        # Ensure list for uniform processing
        if isinstance(truth_passages, str):
            truth_passages_list = [truth_passages]
        else:
            truth_passages_list = list(truth_passages)

        # Normalize truth passage URLs and map to ids
        mapped_pids = []
        for tp in truth_passages_list:
            norm_tp = tp
            if norm_tp is None:
                continue
            # pid = url2id.get(norm_tp)
            pid = norm_tp
            if pid is not None:
                mapped_pids.append(pid)
                continue
            # split_url = tp.split(r"_/")
            # # try these two also
            # pid1 = url2id.get(split_url[0])
            # pid2 = url2id.get(split_url[1])
            # if pid1 is not None:
            #     mapped_pids.append(pid1)
            #     continue
            # if pid2 is not None:
            #     mapped_pids.append(pid2)
                # continue
        # if not mapped_pids:
            # pid = url2id.get(truth_url)
            # if pid is not None:
            #     mapped_pids.append(pid)


        # Remove queries that do not have any truth passage present in
        # the passage collection
        if not mapped_pids:
            continue

        # Build queries exactly as in the snippet: truncate context, reverse,
        # replace newlines and join with " [SEP] ".
        context = trunc_content(context)
        context.reverse()
        user_question = (
            entry["Question"] if int(turn_no) != 1 else entry["Truth_rewrite"]
        )
        user_question = user_question.replace("\n", " ")[: (64 * 5)].strip()
        if context:
            ctx = " [SEP] ".join(context).replace("\n", " ").strip()
            all_user_question = user_question + " [SEP] " + ctx
        else:
            all_user_question = user_question

        if split_name == "test":
            # row-id based queries (as in the rest of this repo)
            queries_row_id_all_test.append((row_id, all_user_question))
            queries_row_id_last_test.append((row_id, user_question))
            human_rewrites_test.append((row_id, entry["Truth_rewrite"]))

            # turn-id based queries to match qrels_qrecc_* (Conversation_Turn)
            queries_turnid_all_test.append((turn_id, all_user_question))
            queries_turnid_last_test.append((turn_id, user_question))

            # qrels keyed by row id (used by existing configs)
            for pid in mapped_pids:
                qrels_test.setdefault(row_id, {})[pid] = 1
            # qrels keyed by Conversation_no_Turn_no (to match snippet)
            for pid in mapped_pids:
                qrels_test_turnid.setdefault(turn_id, {})[pid] = 1
        else:
            queries_row_id_all_train.append((row_id, all_user_question))
            queries_row_id_last_train.append((row_id, user_question))
            human_rewrites_train.append((row_id, entry["Truth_rewrite"]))

            queries_turnid_all_train.append((turn_id, all_user_question))
            queries_turnid_last_train.append((turn_id, user_question))

            for pid in mapped_pids:
                qrels_train.setdefault(row_id, {})[pid] = 1
            for pid in mapped_pids:
                qrels_train_turnid.setdefault(turn_id, {})[pid] = 1

        row_id += 1




with open(f"{base_folder}DATA/qrecc/queries_row_id_dev_all.tsv", "w") as f:
    for qid, query in queries_row_id_all_test:
        f.write(f"{qid}\t{query}\n")

with open(f"{base_folder}DATA/qrecc/queries_row_id_dev_last.tsv", "w") as f:
    for qid, query in queries_row_id_last_test:
        f.write(f"{qid}\t{query}\n")

with open(f"{base_folder}DATA/qrecc/human_rewrites_dev.tsv", "w") as f:
    for qid, rewrite in human_rewrites_test:
        f.write(f"{qid}\t{rewrite}\n")

with open(f"{base_folder}DATA/qrecc/queries_row_id_train_all.tsv", "w") as f:
    for qid, query in queries_row_id_all_train:
        f.write(f"{qid}\t{query}\n")

with open(f"{base_folder}DATA/qrecc/queries_row_id_train_last.tsv", "w") as f:
    for qid, query in queries_row_id_last_train:
        f.write(f"{qid}\t{query}\n")

with open(f"{base_folder}DATA/qrecc/human_rewrites_train.tsv", "w") as f:
    for qid, rewrite in human_rewrites_train:
        f.write(f"{qid}\t{rewrite}\n")

# Turn-id based query files (Conversation_no_Turn_no as in your snippet).
with open(f"{base_folder}DATA/qrecc/queries_turnid_dev_all.tsv", "w") as f:
    for qid, query in queries_turnid_all_test:
        f.write(f"{qid}\t{query}\n")

with open(f"{base_folder}DATA/qrecc/queries_turnid_dev_last.tsv", "w") as f:
    for qid, query in queries_turnid_last_test:
        f.write(f"{qid}\t{query}\n")

with open(f"{base_folder}DATA/qrecc/queries_turnid_train_all.tsv", "w") as f:
    for qid, query in queries_turnid_all_train:
        f.write(f"{qid}\t{query}\n")

with open(f"{base_folder}DATA/qrecc/queries_turnid_train_last.tsv", "w") as f:
    for qid, query in queries_turnid_last_train:
        f.write(f"{qid}\t{query}\n")

with open(f"{base_folder}DATA/qrecc/qrels_rowid_dev.json", "w") as f:
    json.dump(qrels_test, f)

with open(f"{base_folder}DATA/qrecc/qrels_rowid_train.json", "w") as f:
    json.dump(qrels_train, f)

# Also write qrels keyed by Conversation_no_Turn_no, following your snippet.
with open(f"{base_folder}DATA/qrecc/qrels_qrecc_test.json", "w") as f:
    json.dump(qrels_test_turnid, f)

with open(f"{base_folder}DATA/qrecc/qrels_qrecc_train.json", "w") as f:
    json.dump(qrels_train_turnid, f)


# write passages.tsv using the same normalized id space
with open(f"{base_folder}DATA/qrecc/passages.tsv", "w", encoding="utf-8") as f:
    for passage in passages_dataset:
        url = passage["id"]
        # pid = url2id[url]
        pid = url
        contents = passage["contents"]
        safe_contents = contents.replace("\t", " ").replace("\n", " ")
        f.write(f"{pid}\t{safe_contents}\n")

print(f"Total passages written with new ids: {len(passages_dataset)}")
