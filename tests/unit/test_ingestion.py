import pandas as pd
import pytest

from retention_ai.data.ingestion import load_data


def test_load_csv(tmp_path):
    file_path = tmp_path / "sample.csv"

    df = pd.DataFrame({
        "a": [1, 2],
        "b": ["x", "y"]
    })

    df.to_csv(file_path, index=False)

    loaded = load_data(file_path)

    assert loaded.shape == (2, 2)
    assert list(loaded.columns) == ["a", "b"]


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data("does_not_exist.csv")


def test_unsupported_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError):
        load_data(file_path)