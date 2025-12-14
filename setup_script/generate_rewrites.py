import datasets
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForCausalLM, AutoConfig
import gc
import csv
import tqdm



prompt = """# Instruction: I will give you a conversation between a user and a system. You should rewrite the last question of the user into a self-contained query.  

# Context: 
{ctx}

# Please rewrite the following user question: 
{question}

# Re-written query:
"""

def generate_rewrites(questions_loc, qrels_loc, passage_loc, model_name, output_name):
    entries = pd.read_csv(
        questions_loc,
        sep="\t",
        header=None
    )
    if entries.shape[1] == 2:
        entries.columns = ["row_id", "question"]
    elif entries.shape[1] == 1:
        # no explicit row_id column, synthesize it
        entries.columns = ["question"]
        entries = entries.reset_index().rename(columns={"index": "row_id"})
    else:
        raise ValueError(f"Unexpected number of columns in questions file: {entries.shape[1]}")

    # qrels is a mapping from row_id (as string/int) to {doc_id: rel}
    # with open(qrels_loc, "r", encoding="utf-8") as f:
    #     qrels = json.load(f)

    # # load passages into a simple dict
    # passages = {}
    # with open(passage_loc, "r", encoding="utf-8") as f:
    #     reader = csv.reader(f, delimiter="\t")
    #     # optionally skip header if first row looks like a header (non-numeric ID)
    #     first = next(reader, None)
    #     if first:
    #         first_id = first[0]
    #         if first_id.isdigit():
    #             passage_text = "\t".join(first[1:-1])
    #             passages[first_id] = passage_text
    #     for row in reader:
    #         if not row:
    #             continue
    #         doc_id = row[0]
    #         passage_text = "\t".join(row[1:-1])  # last field is title
    #         passages[doc_id] = passage_text

    gc.collect()

    all_prompts = []
    ids = []
    rewrites = {}

    # limit size of conversational context to avoid truncating the final question
    max_ctx_chars = 3000

    # iterate over questions, align via row_id key into qrels
    for _, row in entries.iterrows():
        row_id = str(row["row_id"])
        question = row["question"]
        # if row_id not in qrels:
        #     raise KeyError(f"Row ID {row_id} not found in qrels")
        # qrel_for_row = qrels[row_id]
        # # qrel_for_row looks like {"5498209": 1} or {5498209: 1}
        # doc_id = next(iter(qrel_for_row.keys()))
        # doc_id_str = str(doc_id)
        # if doc_id_str not in passages and doc_id_str.isdigit():
        #     # try int-cast variant if passage IDs are pure ints
        #     doc_id_str = str(int(doc_id_str))
        # if doc_id_str not in passages:
        #     raise KeyError(f"Doc id {doc_id} (row_id={row_id}) not found in passages")

        # passage = passages[doc_id_str]
        # iterate through the questions and generate rewrites
        # expected format per row (example, history in reverse):
        #   2\twhen was the battle fought [SEP] unanswerable [SEP] was the battle fought in australia [SEP] the army personnel ... [SEP] what was australia's contribution ...
        # structure:
        #   current_q [SEP] prev_a [SEP] prev_q [SEP] prevprev_a [SEP] prevprev_q ...
        # we want:
        #   question -> current_q (the FIRST segment)
        #   ctx -> previous history as alternating
        #          "user: <q>\nsystem: <a>" blocks, starting from the
        #          most recent pair (prev_q, prev_a) back in time, but
        #          truncated so that the final question is not lost
        #          during tokenization.
        parts = [p.strip() for p in question.split("[SEP]") if p.strip()]

        if len(parts) == 0:
            # empty line, skip context and question
            ctx_str = ""
            final_question = ""
        else:
            # first segment is the current question (history is reversed)
            final_question = parts[0]

            # build previous history from remaining segments
            # remaining: [prev_a, prev_q, prevprev_a, prevprev_q, ...]
            remaining = parts[1:]
            history_blocks = []
            running_len = 0

            # iterate in steps of 2: (answer, question), (answer, question), ...
            for i in range(0, len(remaining) - 1, 2):
                ans = remaining[i]
                q = remaining[i + 1]
                if not q or not ans:
                    continue
                block = f"user: {q}\\nsystem: {ans}"

                # stop adding history if we would exceed the character budget
                extra_len = len(block) + (1 if history_blocks else 0)
                if running_len + extra_len > max_ctx_chars:
                    break

                history_blocks.append(block)
                running_len += extra_len

            # join all history blocks with newlines
            ctx_str = "\n".join(history_blocks)

        prompt_filled = prompt.format(ctx=ctx_str, question=final_question)
        all_prompts.append(prompt_filled)
        ids.append(row_id)

    if 't5' in model_name.lower():
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-base')
    else:
        # Load and sanitize config for models like Llama 3 whose rope_scaling
        # dict may contain extra keys that older transformers versions reject.
        config = AutoConfig.from_pretrained(model_name)
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and "type" not in rope_scaling:
            factor = float(rope_scaling.get("factor", 1.0))
            config.rope_scaling = {"type": "linear", "factor": factor}

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if 'qwen' in model_name.lower():
            tokenizer.padding_side='left'

        # Ensure the tokenizer has a pad token for batching with padding.
        if tokenizer.pad_token is None:
            # Prefer eos_token if present; otherwise, add a new PAD token.
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
        )

        # Make sure model and tokenizer agree on pad_token_id
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    # Prefer keeping the end of the prompt (which contains the
    # current question and the "# Re-written query:" marker) when
    # we have to truncate.
    tokenizer.truncation_side = "left"
    model.to("cuda")
    batch_size = 16

    def extract_rewrite(generated_text):
        """Extract a single-line rewritten query from the raw model output.

        For causal LMs, the decoded text contains the full prompt, so we
        look for the last occurrence of the marker from the prompt and
        take what follows. For encoder-decoder models (e.g., T5), the
        output is usually just the rewrite; in that case we operate
        directly on the text.
        """

        text = generated_text.strip()

        # If the prompt marker is present (typical for causal LMs), keep
        # only the text after the *last* occurrence to guard against the
        # model echoing the marker again.
        marker = "# Re-written query:"
        idx = text.rfind(marker)
        if idx != -1:
            text = text[idx + len(marker):].strip()

        # Work with the first non-empty line after the marker.
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if not lines:
            return ""

        first_line = lines[0]

        # Strip an optional leading label like "Re-written query: ...".
        lower = first_line.lower()
        if lower.startswith("re-written query"):
            colon_pos = first_line.find(":")
            if colon_pos != -1:
                first_line = first_line[colon_pos + 1 :].strip()

        # Remove surrounding quotes if present.
        if (first_line.startswith("\"") and first_line.endswith("\"")) or (
            first_line.startswith("'") and first_line.endswith("'")
        ):
            first_line = first_line[1:-1].strip()

        return first_line

    for i in tqdm.tqdm(range(0, len(all_prompts), batch_size)):
        batch_prompts = all_prompts[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to("cuda")

        generation_kwargs = dict(
            max_new_tokens=64,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        with torch.no_grad():
            outputs = model.generate(**inputs, **generation_kwargs)

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # first pass extraction
        need_retry_prompts = []
        need_retry_ids = []
        for j, generated_text in enumerate(decoded):
            rewritten_query = extract_rewrite(generated_text)

            # condition to retry: obviously empty or trivially short output
            if (not rewritten_query) or (len(rewritten_query.split()) < 3):
                need_retry_prompts.append(batch_prompts[j])
                need_retry_ids.append(batch_ids[j])
            else:
                rewrites[batch_ids[j]] = rewritten_query

        # retry once for problematic cases with a slightly more exploratory
        # decoding configuration.
        if need_retry_prompts:
            retry_inputs = tokenizer(
                need_retry_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to("cuda")

            retry_generation_kwargs = dict(
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                no_repeat_ngram_size=3,
            )

            with torch.no_grad():
                retry_outputs = model.generate(**retry_inputs, **retry_generation_kwargs)

            retry_decoded = tokenizer.batch_decode(retry_outputs, skip_special_tokens=True)
            for j, generated_text in enumerate(retry_decoded):
                rewritten_query = extract_rewrite(generated_text)
                rewrites[need_retry_ids[j]] = rewritten_query

    # write to output file

    # sort the dict based on keys (row_id). These keys are strings so should be interpreted as integers
    with open(output_name, "w", encoding="utf-8") as f:
        for row_id, rewrite in sorted(rewrites.items(), key=lambda x: int(x[0])):
            f.write(f"{row_id}\t{rewrite}\n")



if __name__ == "__main__":
    # generate_rewrites('/scratch-shared/disco/DATA/topiocqa_topics/queries_rowid_train_all.tsv', '/scratch-shared/disco/DATA/topiocqa_topics/qrel_rowid_train.json', '/scratch-shared/disco/DATA/full_wiki_segments_topiocqa.tsv', 'castorini/t5-base-canard', '/scratch-shared/disco/DATA/topiocqa_rewrites/rewrites_t5_train_last.tsv')
    # generate_rewrites('/scratch-shared/disco/DATA/topiocqa_topics/queries_rowid_train_all.tsv', '/scratch-shared/disco/DATA/topiocqa_topics/qrel_rowid_train.json', '/scratch-shared/disco/DATA/full_wiki_segments_topiocqa.tsv', 'mistralai/Mistral-7B-Instruct-v0.2', '/scratch-shared/disco/DATA/topiocqa_rewrites/rewrites_mistral_train_last.tsv')
    # generate_rewrites('/scratch-shared/disco/DATA/topiocqa_topics/queries_rowid_train_all.tsv', '/scratch-shared/disco/DATA/topiocqa_topics/qrel_rowid_train.json', '/scratch-shared/disco/DATA/full_wiki_segments_topiocqa.tsv', 'meta-llama/Meta-Llama-3.1-8B-Instruct', '/scratch-shared/disco/DATA/topiocqa_rewrites/rewrites_llama_train_last.tsv')
    # generate_rewrites('/scratch-shared/disco/DATA/topiocqa_topics/queries_rowid_dev_all.tsv', '/scratch-shared/disco/DATA/topiocqa_topics/qrel_rowid_dev.json', '/scratch-shared/disco/DATA/full_wiki_segments_topiocqa.tsv', 'meta-llama/Meta-Llama-3.1-8B-Instruct', '/scratch-shared/disco/DATA/topiocqa_rewrites/rewrites_llama_test_last.tsv')
    # generate_rewrites('/scratch-shared/disco/DATA/cast20/queries_rowid_dev_all.tsv', '/scratch-shared/disco/DATA/cast20/qrel_rowid_dev.json', '/scratch-shared/disco/DATA/cast20/passages.tsv', 'meta-llama/Meta-Llama-3-8B-Instruct', '/scratch-shared/disco/DATA/cast20_rewrites/rewrites_qwen_test_last.tsv')
    # generate_rewrites('/scratch-shared/disco/DATA/ikat23/queries_rowid_dev_all.tsv', '/scratch-shared/disco/DATA/ikat23/qrel_rowid_dev.json', '/scratch-shared/disco/DATA/ikat23/passages.tsv', 'castorini/t5-base-canard', '/scratch-shared/disco/DATA/ikat23_rewrites/rewrites_t5_test_last.tsv')
    # generate_rewrites('/scratch-shared/disco/DATA/ikat24/queries_rowid_dev_all.tsv', '/scratch-shared/disco/DATA/ikat24/qrel_rowid_dev.json', '/scratch-shared/disco/DATA/ikat24/passages.tsv', 'mistralai/Mistral-7B-Instruct-v0.2', '/scratch-shared/disco/DATA/ikat24_rewrites/rewrites_mistral_test_last.tsv')
    # generate_rewrites('/scratch-shared/disco/DATA/ikat23/queries_rowid_dev_all.tsv', '/scratch-shared/disco/DATA/ikat23/qrel_rowid_dev.json', '/scratch-shared/disco/DATA/ikat23/passages.tsv', 'mistralai/Mistral-7B-Instruct-v0.2', '/scratch-shared/disco/DATA/ikat23_rewrites/rewrites_mistral_test_last.tsv')
    # generate_rewrites('/scratch-shared/disco/DATA/ikat23/queries_rowid_train_all.tsv', '/scratch-shared/disco/DATA/ikat23/qrel_rowid_train.json', '/scratch-shared/disco/DATA/ikat23/passages.tsv', 'mistralai/Mistral-7B-Instruct-v0.2', '/scratch-shared/disco/DATA/ikat23_rewrites/rewrites_mistral_train_last.tsv')
    generate_rewrites('/scratch-shared/disco/DATA/ikat24/queries_rowid_dev_all.tsv', '/scratch-shared/disco/DATA/ikat24/qrel_rowid_dev.json', '/scratch-shared/disco/DATA/ikat24/passages.tsv', 'castorini/t5-base-canard', '/scratch-shared/disco/DATA/ikat24_rewrites/rewrites_t5_test_last.tsv')
