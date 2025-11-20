# # train set qrecc


# download huggingface dataset slupart/qrecc
import json
from datasets import load_dataset
dataset = load_dataset("slupart/qrecc")


qrels_test = {}
qrels_train = {}

queries_row_id_all_test = []
queries_row_id_all_train = []
queries_row_id_last_test = []
queries_row_id_last_train = []

row_id = 0

def create_url2id_mapping(dataset):
    url2id = {}
    ids = 0
    for split in ["train", "test"]:
        for entry in dataset[split]:
            url2id[entry['Answer_URL']] = ids
            ids += 1
    return url2id


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

url2id = create_url2id_mapping(dataset)
for split in ["train", "test"]:
    for entry in dataset[split]:
        conv_no = entry["Conversation_no"]
        turn_no = entry["Turn_no"]
        context = entry["Context"]
        rewrite = entry["Truth_rewrite"]
        answer = entry["Truth_answer"]
        # Truth_passages contains URLs; we need to map them to integer passage ids
        # using the url2id mapping provided by the qrecc-passages dataset
        truth_passages = entry["Answer_URL"]
        
        context.reverse()
        user_question = entry["Question"] if turn_no != 1 else entry["Truth_rewrite"]
        context = trunc_content(context)
        user_question = user_question.replace("\n", " ")[: (64 * 5)].strip()
        if context:
            all_user_question = user_question + " [SEP] " + " [SEP] ".join(context)
        else:
            all_user_question = user_question

        if split == "test":
            queries_row_id_all_test.append((row_id, all_user_question))
            queries_row_id_last_test.append((row_id, user_question))
            human_rewrites_test.append((row_id, entry["Truth_rewrite"]))
            passage_id = url2id[truth_passages]
            qrels_test.setdefault(row_id, {})[passage_id] = 1
        else:
            queries_row_id_all_train.append((row_id, all_user_question))
            queries_row_id_last_train.append((row_id, user_question))
            human_rewrites_train.append((row_id, entry["Truth_rewrite"]))
            passage_id = url2id[truth_passages]
            qrels_train.setdefault(row_id, {})[passage_id] = 1

        row_id += 1

import os
os.makedirs("DATA/qrecc", exist_ok=True)

with open("DATA/qrecc/queries_row_id_dev_all.tsv", "w") as f:
    for row_id, query in queries_row_id_all_test:
        f.write(f"{row_id}\t{query}\n")
with open("DATA/qrecc/queries_row_id_dev_last.tsv", "w") as f:
    for row_id, query in queries_row_id_last_test:
        f.write(f"{row_id}\t{query}\n")
with open("DATA/qrecc/human_rewrites_dev.tsv", "w") as f:
    for row_id, rewrite in human_rewrites_test:
        f.write(f"{row_id}\t{rewrite}\n")
with open("DATA/qrecc/queries_row_id_train_all.tsv", "w") as f:
    for row_id, query in queries_row_id_all_train:
        f.write(f"{row_id}\t{query}\n")
with open("DATA/qrecc/queries_row_id_train_last.tsv", "w") as f:
    for row_id, query in queries_row_id_last_train:
        f.write(f"{row_id}\t{query}\n")
with open("DATA/qrecc/human_rewrites_train.tsv", "w") as f:
    for row_id, rewrite in human_rewrites_train:
        f.write(f"{row_id}\t{rewrite}\n")
with open("DATA/qrecc/qrels_rowid_dev.json", "w") as f:
    json.dump(qrels_test, f)
with open("DATA/qrecc/qrels_rowid_train.json", "w") as f:
    json.dump(qrels_train, f)


dataset = load_dataset("slupart/qrecc-passages", split="train")

buffer = []
buffer_size = 10_000

not_found = 0
last_id = len(url2id)
with open("DATA/qrecc/passages.tsv", "w") as f:
    for entry in dataset:
        if entry['id'] not in url2id:
            not_found += 1
            new_id = last_id
            last_id += 1
            url2id[entry['id']] = new_id
        else:
            new_id = url2id[entry['id']]
        buffer.append(f"{new_id}\t{entry['contents']}\n")
        if len(buffer) >= buffer_size:
            f.writelines(buffer)
            buffer = []

    # write remaining lines
    if buffer:
        f.writelines(buffer)

print(f"Number of passages not found in url2id mapping: {not_found}")