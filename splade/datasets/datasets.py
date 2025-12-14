import os
from torch.utils.data import Dataset
from tqdm.auto import tqdm


CHUNK_SIZE = 4096


def _process_chunk(args):
    """Worker helper: parse a chunk of lines and return list of (mode, id_, data)."""
    lines_chunk, id_style, topiocqa = args
    results = []

    for line, i in lines_chunk:
        if len(line) <= 1:
            continue

        if topiocqa:
            if i == 0:
                # header line
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            id_, text, title = parts
            id_ = id_.strip()
            data = f"{title}. {text}"
        else:
            # only need two fields: id and text
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            id_, data = parts
            id_ = id_.strip()
            # keep semantics: replace newlines with spaces
            data = data.replace("\n", " ").replace("\r", " ")

        data = data.strip()
        if id_style == "row_id":
            results.append(("row_id", id_, data))
        else:
            results.append(("content_id", id_, data))

    return results


class CollectionDatasetPreLoad(Dataset):
    """Dataset to iterate over a document/query collection.

    Format per line: ``doc_id \t doc``. The whole collection is preloaded in
    memory at init. When used in a distributed setup (e.g. with ``torchrun``),
    you can pass ``rank`` and ``world_size`` so that each process only loads a
    disjoint shard of the collection based on the original line order.
    """

    def __init__(self, data_dir, id_style, max_sample=None, topiocqa=False, rank: int = 0, world_size: int = 1):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        # id_style indicates how we access the doc/q (row id or doc/q id)
        self.id_style = id_style
        self.data_dict = {}
        self.line_dict = {}
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))

        print(f"Preloading dataset (rank {self.rank}/{self.world_size})")
        curr_id = 0
        if ".tsv" not in self.data_dir:
            path_collection = os.path.join(self.data_dir, "raw.tsv")
        else:
            path_collection = self.data_dir

        # Read file and build chunks of lines (I/O bound, single process)
        chunks = []
        current_chunk = []
        with open(path_collection) as reader:
            for i, line in enumerate(reader):
                # Shard lines across ranks in a deterministic way so that each
                # process only sees approximately 1/world_size of the
                # collection. This avoids duplicated work when using DDP.
                if i % self.world_size != self.rank:
                    continue

                if max_sample and i > max_sample:
                    break
                current_chunk.append((line, i))
                if len(current_chunk) >= CHUNK_SIZE:
                    chunks.append(current_chunk)
                    current_chunk = []
        if current_chunk:
            chunks.append(current_chunk)

        # Sequential processing of chunks. With DDP sharding each rank only
        # sees a subset of the collection, so this keeps things simple and
        # avoids multiprocessing issues when spawned under torchrun.
        for chunk in tqdm(chunks, total=len(chunks)):
            results = _process_chunk((chunk, self.id_style, topiocqa))
            for mode, id_, data in results:
                if mode == "row_id":
                    # Preserve the original behavior: row_id keys are 0..N-1 in reading order
                    self.data_dict[curr_id] = data
                    self.line_dict[curr_id] = id_
                    curr_id += 1
                else:
                    self.data_dict[id_] = data

        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]