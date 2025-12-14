import json
from collections import defaultdict

def combine(distil_paths, output_path):
    # a dict of dicts with lists
    combined_scores = defaultdict(lambda: defaultdict(list))
    for path in distil_paths:
        with open(path, 'r') as f:
            data = json.load(f)
            # append to list of scores for each query-document pair
            for qid, doc_scores in data.items():
                for doc_id, score in doc_scores.items():
                    combined_scores[qid][doc_id].append(score)

    # average the scores
    averaged_scores = {}
    for qid, doc_scores in combined_scores.items():
        averaged_scores[qid] = {}
        for doc_id, scores in doc_scores.items():
            averaged_scores[qid][doc_id] = sum(scores) / len(scores)
    
    with open(output_path, 'w') as f:
        json.dump(averaged_scores, f)

if __name__ == "__main__":
    distil_paths = [
        '/scratch-shared/disco/DATA/topiocqa_distil/distil_run_top_mistral.json',
        '/scratch-shared/disco/DATA/topiocqa_distil/distil_run_top_t5.json',
    ]
    output_path = '/scratch-shared/disco/DATA/topiocqa_distil/distil_run_top_multi.json'

    combine(distil_paths, output_path)