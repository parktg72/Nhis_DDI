import hashlib


def _make_trainer():
    from scripts.train.trainer import XGBoostTrainer
    t = XGBoostTrainer.__new__(XGBoostTrainer)
    t.model = None  # None is picklable; MagicMock is not
    t.params = {}
    t.feature_importances_ = None
    t.best_threshold_ = 0.5
    t._trained = True
    t.config = None
    return t


def test_save_generates_sha256_sidecar(tmp_path):
    """save() must create a .sha256 sidecar file."""
    t = _make_trainer()
    path = tmp_path / "model.pkl"
    t.save(path)

    sha_path = path.with_suffix(".pkl.sha256")
    assert sha_path.exists(), ".sha256 sidecar missing"

    content = path.read_bytes()
    expected = hashlib.sha256(content).hexdigest()
    actual = sha_path.read_text().strip().split()[0]
    assert actual == expected, "SHA-256 mismatch"


def test_save_sidecar_content_matches_file(tmp_path):
    """Sidecar content must match actual file bytes."""
    t = _make_trainer()
    path = tmp_path / "model2.pkl"
    t.save(path)
    sha_path = path.with_suffix(".pkl.sha256")
    stored_hash = sha_path.read_text().strip().split()[0]
    computed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert stored_hash == computed
