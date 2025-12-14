# Download and extract the topics files for the TopiocQA dataset
mkdir -p DATA/topiocqa_topics/
wget -O DATA/topiocqa_topics/raw_train.json https://zenodo.org/records/6151011/files/data/retriever/all_history/train.json?download=1
wget -O DATA/topiocqa_topics/raw_dev.json https://zenodo.org/records/6151011/files/data/retriever/all_history/dev.json?download=1

# Download and extract the collection files for the TopiocQA dataset
wget -O DATA/full_wiki_segments_topiocqa.tsv https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv?download=1

wget -O DATA/topiocqa_rewrites/rewrites_human.json https://zenodo.org/records/6151011/files/data/gold_passages_info/rewrites_t5_qrecc/dev.json

mkdir -p DATA/ikat24/
wget -O DATA/ikat24/2024_test_topics.json https://raw.githubusercontent.com/irlabamsterdam/iKAT/refs/heads/main/2024/data/2024_test_topics.json

mkdir -p DATA/ikat23/
wget -O DATA/ikat23/2023_test_topics.json https://raw.githubusercontent.com/irlabamsterdam/iKAT/refs/heads/main/2023/data/2023_test_topics.json

mkdir -p DATA/qrecc/

