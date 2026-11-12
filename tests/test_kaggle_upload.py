"""kaggle/make_upload.py ships nltk_data as one deterministic tar (Kaggle extracts every .zip on upload)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("make_upload", ROOT / "kaggle" / "make_upload.py")
mu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mu)


def test_tar_deterministic(tmp_path):
    src = tmp_path / "nd"
    (src / "corpora").mkdir(parents=True)
    (src / "corpora" / "x.zip").write_bytes(b"abc")
    a = mu.nltk_tar(src, tmp_path / "a.tar")
    (src / "corpora" / "x.zip").touch()  # mtime changes, content does not
    assert mu.nltk_tar(src, tmp_path / "b.tar") == a


@pytest.mark.skipif(not (ROOT / "nltk_data").exists(), reason="repo nltk_data/ (gitignored) not present")
def test_repo_tar_matches_setup_pin(tmp_path):
    assert mu.nltk_tar(ROOT / "nltk_data", tmp_path / "n.tar") == mu.NLTK_TAR_SHA256
