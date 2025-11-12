"""Create baseline rewritten query histories in the desired Q / (A,Q) reverse chain.

Requested target format for each row i (0-indexed):
    current_question_i [SEP] answer_{i-1} [SEP] question_{i-1} [SEP] answer_{i-2} [SEP] question_{i-2} ...

So we always begin the row with the *current* question rewrite, then append
alternating (answer, question) pairs for all PREVIOUS turns in reverse
chronological order. The very first row (i=0) contains only its question.

We mine answer spans from the original dev topic TSV:
    DATA/topiocqa_topics/queries_rowid_dev_all.tsv

Heuristic (unchanged): For row i in the original TSV, its answer appears in row i+1
between the first and second [SEP] tokens. Example lines:
    0    Q0
    1    Q1 [SEP] A0 [SEP] Q0           => answer for Q0 is A0
If the next line doesn't have at least two [SEP] tokens we record empty answer.

Outputs (updated):
1. DATA/topiocqa_rewrites/rewrites_mistral_test_all.tsv (id, chained_history)
2. DATA/topiocqa_rewrites/rewrites_mistral_test_first_only.tsv (id, first_entry_only)
   - Contains only the first entry of each row (the actual last/current question in the chat)
"""

from pathlib import Path
import pandas as pd

# Input parquet with rewritten queries
REWRITES_PARQUET = Path('DATA/topiocqa_rewrites/rewrites_mistral_test.parquet')
ORIG_TOPICS_TSV = Path('DATA/topiocqa_topics/queries_rowid_dev_all.tsv')

def build_histories(df: pd.DataFrame, answers: dict):
    """Build the desired chained history format.

    For each row i we emit:
        Q_i [SEP] A_{i-1} [SEP] Q_{i-1} [SEP] A_{i-2} [SEP] Q_{i-2} ...

    Answers dict maps ORIGINAL numeric row id -> answer string (may be empty).
    We parse conversation ID from the 'id' column (format: "conv_turn") and
    reset history when a new conversation starts.
    
    Returns:
        ids: list of ID strings
        texts: list of full chained history strings
        row_ids: list of original row IDs
        first_entries: list of first entries only (current question)
    """
    ids = []
    texts = []
    row_ids = []  # Track original row IDs
    first_entries = []  # Track only the first entry (current question)
    # Cache prior questions & answers as we iterate
    prior_q = []  # list of question strings
    prior_a = []  # list of answer strings aligned with prior_q indices
    current_conv = None
    
    for idx, row in df.iterrows():
        # Parse conversation ID from 'id' column (format: "conv_turn")
        id_str = str(row['id'])
        conv_id = id_str.split('_')[0] if '_' in id_str else id_str
        
        # If new conversation, reset history
        if conv_id != current_conv:
            current_conv = conv_id
            prior_q = []
            prior_a = []
        
        q_text = str(row['text']).strip()
        chain_parts = [q_text]
        # Append reverse (A,Q) pairs for all previous turns in this conversation
        for j in range(len(prior_q) - 1, -1, -1):
            a_j = prior_a[j]
            # Always include the answer (even if empty) and question
            if a_j:
                chain_parts.append(a_j)
            chain_parts.append(prior_q[j])
        joined = ' [SEP] '.join(chain_parts)
        ids.append(id_str)
        texts.append(joined)
        row_ids.append(idx)
        first_entries.append(q_text)  # Store only the current question
        # Add current question & its mined answer for next iterations
        prior_q.append(q_text)
        prior_a.append(answers.get(idx, '').strip())
    return ids, texts, row_ids, first_entries

def mine_answers(tsv_path: Path):
    """Mine answers from the original topics TSV.

    Returns dict: question_row_id -> answer string.
    """
    answers = {}
    with tsv_path.open('r') as f:
        lines = f.readlines()
    # Parse each line into (row_id, text)
    parsed = []
    for line in lines:
        line = line.rstrip('\n')
        if not line:
            continue
        try:
            id_part, text_part = line.split('\t', 1)
        except ValueError:
            # If malformed just skip
            continue
        try:
            row_id = int(id_part)
        except ValueError:
            # Non-integer id skip
            continue
        parsed.append((row_id, text_part))
    # Sort to ensure order (though file should already be ordered)
    parsed.sort(key=lambda x: x[0])
    for i in range(len(parsed) - 1):
        curr_id, _curr_text = parsed[i]
        _next_id, next_text = parsed[i + 1]
        parts = next_text.split('[SEP]')
        # Need at least 2 SEP tokens => 3 parts (question, answer, ...)
        if len(parts) >= 3:
            answer = parts[1].strip()
        else:
            answer = ''
        answers[curr_id] = answer
    # Last question has no subsequent line to mine answer from
    last_id, _ = parsed[-1]
    answers.setdefault(last_id, '')
    return answers

def main():
    if not REWRITES_PARQUET.exists():
        raise FileNotFoundError(f"Missing rewrites parquet: {REWRITES_PARQUET}")
    df = pd.read_parquet(REWRITES_PARQUET)

    # Mine answers first so we can use them while building histories
    if ORIG_TOPICS_TSV.exists():
        answers = mine_answers(ORIG_TOPICS_TSV)
    else:
        answers = {}

    ids, texts, row_ids, first_entries = build_histories(df, answers)

    # Write updated two-column TSV (id, chained_history)
    out_simple = Path('DATA/topiocqa_rewrites/rewrites_mistral_test_all.tsv')
    with out_simple.open('w') as f:
        for i, txt in zip(row_ids, texts):
            f.write(f"{i}\t{txt}\n")

    # Write TSV with only first entries (current questions)
    out_first_only = Path('DATA/topiocqa_rewrites/rewrites_mistral_test_last.tsv')
    with out_first_only.open('w') as f:
        for i, first_entry in zip(row_ids, first_entries):
            f.write(f"{i}\t{first_entry}\n")

    
            

if __name__ == '__main__':
    main()