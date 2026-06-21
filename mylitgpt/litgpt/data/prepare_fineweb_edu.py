# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import os
import time
import traceback
from functools import partial
from pathlib import Path

from litgpt.constants import _LITDATA_AVAILABLE
from litgpt.tokenizer import Tokenizer
from litgpt.utils import CLI, extend_checkpoint_dir

if _LITDATA_AVAILABLE:
    from litdata.processing.data_processor import DataChunkRecipe
else:
    DataChunkRecipe = object


class FineWebEduDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(self, tokenizer: Tokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer

    def prepare_structure(self, input_dir):
        files = Path(input_dir).rglob("*.parquet")
        return [str(file) for file in files]

    def prepare_item(self, item_metadata):
        import pyarrow.parquet as pq

        filepath = item_metadata
        start = time.time()

        try:
            parquet_file = pq.ParquetFile(filepath)
            for batch in parquet_file.iter_batches(batch_size=8192, columns=["text"]):
                for text in batch.to_pandas()["text"]:
                    if not text:
                        continue
                    yield self.tokenizer.encode(text, bos=False, eos=True)

        except Exception:
            print(traceback.format_exc())
            print(f"Error reading {filepath}")
            return

        parquet_file.close()
        end = time.time()
        print(f"Took {end - start:.2f} seconds total", filepath)


def tokenize(filepath: str, tokenizer: Tokenizer):
    yield from FineWebEduDataRecipe(tokenizer=tokenizer, chunk_size=1).prepare_item(filepath)


def prepare(
    input_dir: Path = Path("data/fineweb-edu/sample-10BT"),
    output_dir: Path = Path("data/fineweb-edu-10bt/qwen25/train"),
    tokenizer_path: Path = Path("checkpoints/Qwen/Qwen2.5-0.5B/"),
    chunk_size: int = (2049 * 8192),
    fast_dev_run: bool = False,
) -> None:
    from litdata import TokensLoader, optimize

    tokenizer_path = extend_checkpoint_dir(tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = FineWebEduDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    files = data_recipe.prepare_structure(input_dir)
    if not files:
        raise FileNotFoundError(f"No parquet files found under {input_dir}")
    if fast_dev_run:
        files = files[:1]

    start_time = time.time()
    optimize(
        fn=partial(tokenize, tokenizer=tokenizer),
        inputs=files,
        output_dir=str(output_dir),
        num_workers=1 if fast_dev_run else min(len(files), os.cpu_count() or 1),
        chunk_bytes="3GB",
        item_loader=TokensLoader(),
    )
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    CLI(prepare)
