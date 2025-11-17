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

for split in ["train", "test"]:
    for entry in dataset[split]:
        conv_no = entry["Conversation_no"]
        turn_no = entry["Turn_no"]
        context = entry["Context"]
        rewrite = entry["Truth_rewrite"]
        answer = entry["Truth_answer"]
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
            for passage in entry["Truth_passages"]:
                qrels_test.setdefault(row_id, {})[passage] = 1
        else:
            queries_row_id_all_train.append((row_id, all_user_question))
            queries_row_id_last_train.append((row_id, user_question))
            human_rewrites_train.append((row_id, entry["Truth_rewrite"]))
            for passage in entry["Truth_passages"]:
                qrels_train.setdefault(row_id, {})[passage] = 1

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

with open("DATA/qrecc/passages.tsv", "w") as f:
    for entry in dataset:
        buffer.append(f"{entry['id']}\t{entry['contents']}\n")
        if len(buffer) >= buffer_size:
            f.writelines(buffer)
            buffer = []

    # write remaining lines
    if buffer:
        f.writelines(buffer)

# qrels = {}
# notin=0
# rowid=0
# with open("/gpfs/work4/0/prjs0871/qrecc/scai/test.json", "r") as f:
#     qrecc_qrel = json.load(f)
#     with open("/gpfs/work4/0/prjs0871/qrecc/scai/rowid_queries_human_qrecc_test.tsv", "w") as tsv_queries_human:
#         with open("/gpfs/work4/0/prjs0871/qrecc/scai/rowid_queries_raw_qrecc_test.tsv", "w") as tsv_queries:
#             with open("/gpfs/work4/0/prjs0871/qrecc/scai/rowid_queries_all_qrecc_test.tsv", "w") as tsv_queries_all:
#                 for topic_turn in qrecc_qrel:
#                     turn_id = str(topic_turn["Conversation_no"])+"_"+str(topic_turn["Turn_no"])
#                     ctx_conv = topic_turn["Context"]
#                     ctx_conv_shorten = []
#                     for i, c in enumerate(ctx_conv):
#                         if i%2==0:
#                             ctx_conv_shorten.append(c[:(64*5)].strip())
#                         else:
#                             ctx_conv_shorten.append(c[:(100*5)].strip())
#                     ctx_conv = ctx_conv_shorten
#                     ctx_conv.reverse()
#                     # ctx_conv = [c[:(64*5)].strip() for c in ctx_conv]
#                     ctx = " [SEP] ".join(ctx_conv).replace("\n", " ").strip()
#                     user_question = topic_turn["Question"] if int(topic_turn['Turn_no']) != 1 else topic_turn["Truth_rewrite"]
#                     user_question = user_question.replace("\n", " ")[:(64*5)].strip()
#                     all_user_question = user_question + " [SEP] " + ctx
#                     tsv_queries.write(str(rowid)+"\t"+user_question+"\n")
#                     tsv_queries_all.write(str(rowid)+"\t"+all_user_question+"\n")
#                     tsv_queries_human.write(str(rowid)+"\t"+topic_turn["Truth_rewrite"]+"\n")

#                     # ctx = topic_turn["Question"].strip()

#                     # all_user_question = topic_turn["question"].split("[SEP]")
#                     # all_user_question.reverse()
#                     # tsv_queries.write(turn_id+"\t"+user_question+"\n")
#                     # tsv_queries_all.write(turn_id+"\t"+"[SEP]".join(all_user_question)+"\n")
#                     # if topic_turn["Answer_URL"] in url2id:
#                     #     qrels[turn_id] = {url2id[topic_turn["Answer_URL"]]: 1}
#                     for t in topic_turn["Truth_passages"]:
#                         if t in url2id:
#                             qrels[turn_id] = {url2id[t]: 1}
#                         else:
#                             notin+=1
#                     rowid+=1
# print(len(qrecc_qrel))
# print(len(qrels))
# print(notin)
# json.dump(qrels, open("/gpfs/work4/0/prjs0871/qrecc/scai/qrels_qrecc_test.json", "w"))