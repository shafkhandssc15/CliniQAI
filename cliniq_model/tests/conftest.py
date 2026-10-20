import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Small but real training run (about 10 s) shared by all engine tests."""
    from cliniq_ml.engine import CliniQEngine, train
    from cliniq_ml.synth import generate
    d = tmp_path_factory.mktemp("model")
    data = generate(2500, 6, seed=21)
    res = train(data, d, seed=5, verbose=False)
    return CliniQEngine(d), res, d
