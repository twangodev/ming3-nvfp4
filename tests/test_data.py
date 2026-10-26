from contextlib import closing

import pyarrow as pa
import pyarrow.parquet as pq

from ming3_nvfp4.data import fleurs_examples


def test_local_parquet_reader_is_deterministic_and_closes_on_partial_iteration(tmp_path):
    path = tmp_path / "examples.parquet"
    pq.write_table(pa.table({"id": list(range(200)), "audio": [b"wave"] * 200}), path)
    first = list(fleurs_examples(path, 42))
    assert first == list(fleurs_examples(path, 42))
    assert sorted(row["id"] for row in first) == list(range(200))
    assert first != list(fleurs_examples(path, 7))
    iterator = fleurs_examples(path, 42)
    with closing(iterator):
        assert next(iterator) == first[0]
    assert iterator.gi_frame is None
