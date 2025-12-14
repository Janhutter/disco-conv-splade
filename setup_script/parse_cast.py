import ir_datasets
import os
import gc
import json

base_folder = '/scratch-shared/disco/'

def parse(dataset, dataset_name):
    # Preload docs into memory: doc_id -> text
    docs_text = {doc.doc_id: doc.text for doc in dataset.docs_iter()}

    print(f"Loaded {len(docs_text)} documents into memory.")

    # Pre-size lists instead of using dicts with sorted keys
    conversations_all = []
    conversations_last = []
    human_rewrites = []
    t5_rewrite = []

    # map original dataset query ids to 0-based row ids used in our files
    queryid2rowid = {}

    curr_conversation_id = None
    curr_conversation_turns = []

    for row_id, query in enumerate(dataset.queries_iter()):
        conv_id = query.topic_number
        if conv_id != curr_conversation_id:
            curr_conversation_id = conv_id
            curr_conversation_turns = []

        raw_utterance = query.raw_utterance
        human_rewrite = query.manual_rewritten_utterance
        t5_rewrite_ = query.automatic_rewritten_utterance
        answer_id = query.manual_canonical_result_id

        # use query.query_id (string) as key to later align qrels with row ids
        qid = str(query.query_id)
        queryid2rowid[qid] = row_id

        answer = docs_text.get(answer_id, "")
        # truncate answer if too long (bug in original: mismatched thresholds, preserved here)
        if len(answer) > 100 * 5:
            answer = answer[: (64 * 5)].rsplit(" ", 1)[0]
        if len(raw_utterance) > 64 * 5:
            raw_utterance = raw_utterance[: (100 * 5)].rsplit(" ", 1)[0]

        history_turns = curr_conversation_turns + [raw_utterance]
        conversations_all.append(" [SEP] ".join(reversed(history_turns)))

        conversations_last.append(raw_utterance)
        human_rewrites.append(human_rewrite)
        t5_rewrite.append(t5_rewrite_)

        curr_conversation_turns.append(raw_utterance)
        curr_conversation_turns.append(answer)

    # outputs identical as before (row ids are 0..n-1 in order)
    os.makedirs(f"{base_folder}DATA/{dataset_name}", exist_ok=True)
    with open(f"{base_folder}DATA/{dataset_name}/queries_rowid_dev_all.tsv", "w", encoding="utf-8") as f:
        for row_id, text in enumerate(conversations_all):
            f.write(f"{row_id}\t{text}\n")
    with open(f"{base_folder}DATA/{dataset_name}/queries_rowid_dev_last.tsv", "w", encoding="utf-8") as f:
        for row_id, text in enumerate(conversations_last):
            f.write(f"{row_id}\t{text}\n")
    with open(f"{base_folder}DATA/{dataset_name}/human_rewrites_dev.tsv", "w", encoding="utf-8") as f:
        for row_id, text in enumerate(human_rewrites):
            f.write(f"{row_id}\t{text}\n")
    with open(f"{base_folder}DATA/{dataset_name}/t5_rewrites_dev.tsv", "w", encoding="utf-8") as f:
        for row_id, text in enumerate(t5_rewrite):
            f.write(f"{row_id}\t{text}\n")

    text2id = {}
    entries = 0

    # Write passages and build mapping in a deterministic order with numeric row IDs
    with open(f"{base_folder}DATA/{dataset_name}/passages.tsv", "w", encoding="utf-8") as f:
        for doc_id in sorted(docs_text.keys()):
            text = docs_text[doc_id].replace("\n", " ").replace("\r", " ").strip()
            f.write(f"{entries}\t{text}\n")
            text2id[doc_id] = entries
            entries += 1

    found = {}
    qrels_dict = {}

    for qrel in dataset.qrels_iter():
        qid = str(qrel.query_id)
        # some qrels may refer to queries that are not in our dev split
        if qid not in queryid2rowid:
            continue

        row_id = queryid2rowid[qid]
        doc_id = qrel.doc_id
        relevance = qrel.relevance
        passage_id = text2id.get(doc_id, None)
        # should only keep a single highest relevance passage per query
        already_in_relevance = found.get(row_id, 0)

        if passage_id is not None and relevance > already_in_relevance:
            qrels_dict[str(row_id)] = {str(passage_id): relevance}
            found[row_id] = relevance

    # it should only write a relevance of 1 per query, as in the original code
    for row_id in qrels_dict:
        for pid in qrels_dict[row_id]:
            qrels_dict[row_id][pid] = 1

    with open(f"{base_folder}DATA/{dataset_name}/qrels_rowid_dev.json", "w", encoding="utf-8") as f:
        json.dump(qrels_dict, f)


# dataset2020 = ir_datasets.load("trec-cast/v1/2020")
# parse(dataset2020, "cast20")
# del dataset2020
# gc.collect()

dataset2019 = ir_datasets.load("trec-cast/v1/2020")
parse(dataset2019, "cast20")
del dataset2019
gc.collect()