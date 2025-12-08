import json
import pandas as pd
from typing import Dict, Any

def load_docs(path: str) -> pd.DataFrame:
    """Load documents TSV (has header row)."""
    return pd.read_csv(path, sep="\t")

def load_queries(path: str) -> pd.DataFrame:
    """Load queries TSV (no header row)."""
    return pd.read_csv(path, sep="\t", header=None, names=["id", "query"])

def build_doc_lookup(df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    """Create mapping: doc_id → {'text':..., 'title':...}."""
    return {int(row.id): {"text": row.text, "title": row.title} for _, row in df.iterrows()}

def process_top_n(
    n: int,
    queries_path: str,
    docs_path: str,
    sim_json_path: str,
    output_path: str
) -> None:
    """
    For the first n queries, determine highest and lowest similarity documents and
    save a JSON list containing, for each query: the query text, the best-scoring
    doc, and the worst-scoring doc.
    """
    queries_df = load_queries(queries_path)
    docs_df = load_docs(docs_path)
    doc_lookup = build_doc_lookup(docs_df)

    with open(sim_json_path) as f:
        sim_data: Dict[str, Dict[str, float]] = json.load(f)

    results = []

    for i in range(n):
        qid = str(i)
        if qid not in sim_data:
            continue

        scores = sim_data[qid]
        high_doc = max(scores, key=scores.get)
        low_doc  = min(scores, key=scores.get)

        query_text = queries_df.loc[i, "query"]
        high_entry = doc_lookup.get(int(high_doc), {})
        low_entry  = doc_lookup.get(int(low_doc), {})

        results.append({
            "query_id": i,
            "query": query_text,
            "highest_sim": {"doc_id": int(high_doc), **high_entry},
            "lowest_sim": {"doc_id": int(low_doc), **low_entry}
        })

    with open(output_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_top_n(
        n=3,
        queries_path="DATA/topiocqa_topics/queries_rowid_dev_all.tsv",
        docs_path="DATA/full_wiki_segments_topiocqa.tsv",
        sim_json_path="DATA/topiocqa_distil/distil_run_top_mistral.json",
        output_path="top_docs_negatives_inspection.json"
    )
