import os
import json


os.makedirs("DATA/topiocqa_subset", exist_ok=True)
# only write the first 10 lines
with open("DATA/topiocqa_topics/queries_rowid_dev_all.tsv") as f:
    with open("DATA/topiocqa_subset/queries_rowid_dev_all.tsv", "w") as out_f:
        for i, line in enumerate(f):
            if i >= 381:
                break
            out_f.write(line)

with open("DATA/topiocqa_topics/queries_rowid_train_all.tsv") as f:
    with open("DATA/topiocqa_subset/queries_rowid_train_all.tsv", "w") as out_f:
        for i, line in enumerate(f):
            if i >= 480:
                break
            out_f.write(line)

with open("DATA/topiocqa_topics/qrel_rowid_dev.json") as f:
    with open("DATA/topiocqa_subset/qrel_rowid_dev.json", "w") as out_f:
        # only write the qrels for the written queries
        qrels = json.load(f)
        subset_qrels = {k: v for i, (k, v) in enumerate(qrels.items()) if i < 381}
        json.dump(subset_qrels, out_f, indent=2)

with open("DATA/topiocqa_topics/qrel_rowid_train.json") as f:
    with open("DATA/topiocqa_subset/qrel_rowid_train.json", "w") as out_f:
        # only write the qrels for the written queries
        qrels = json.load(f)
        subset_qrels = {k: v for i, (k, v) in enumerate(qrels.items()) if i < 480}
        json.dump(subset_qrels, out_f, indent=2)

# write the documents that are in the qrels only
with open("DATA/full_wiki_segments_topiocqa.tsv") as f:
    docs = {}
    for line in f:
        doc_id, content = line.strip().split("\t", 1)
        docs[doc_id] = content

    with open("DATA/topiocqa_subset/full_wiki_segments_topiocqa.tsv", "w") as out_f:
        written_doc_ids = set()
        for qrel_file in ["DATA/topiocqa_subset/qrel_rowid_dev.json", "DATA/topiocqa_subset/qrel_rowid_train.json"]:
            with open(qrel_file) as qf:
                qrels = json.load(qf)
                for doc_ids in qrels.values():
                    for doc_id in doc_ids:
                        if doc_id not in written_doc_ids and doc_id in docs:
                            out_f.write(f"{doc_id}\t{docs[doc_id]}\n")
                            written_doc_ids.add(doc_id)

        len_orig = len(docs)
        len_new = len(written_doc_ids)
        entry = 0
        if len_new < 0.01 * len_orig:
            # add distractor documents to make up at least 1% of the original documents
            for doc_id, content in docs.items():
                if entry == 0:
                    entry += 1
                    continue
                if len_new >= 0.01 * len_orig:
                    break
                if doc_id not in written_doc_ids:
                    out_f.write(f"{doc_id}\t{content}\n")
                    written_doc_ids.add(doc_id)
                    len_new += 1