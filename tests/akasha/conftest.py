from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from akasha.bench.correctness import canonical_init  # noqa: E402
from akasha.config import TRAINER_PATH  # noqa: E402
from akasha.models.arma.config import (  # noqa: E402
    ArmAConfig,
    production_config,
    tiny_config,
)


def pytest_addoption(parser):
    parser.addoption(
        "--skip-slow",
        action="store_true",
        default=False,
        help="skip the 1024/2048 production parity cases",
    )


@pytest.fixture(scope="session")
def slow_enabled(pytestconfig) -> bool:
    return not pytestconfig.getoption("--skip-slow")


@pytest.fixture(scope="session")
def tiny_cfg() -> ArmAConfig:
    return tiny_config()


@pytest.fixture(scope="session")
def prod_cfg() -> ArmAConfig:
    return production_config()


@pytest.fixture(scope="session")
def tiny_weights(tiny_cfg):
    return canonical_init(tiny_cfg)


@pytest.fixture(scope="session")
def prod_weights(prod_cfg):
    return canonical_init(prod_cfg)


@pytest.fixture(scope="session")
def trainer_module():
    """Import the frozen trainer source by path (no repo imports needed)."""
    path = ROOT / TRAINER_PATH
    spec = importlib.util.spec_from_file_location("akasha_trainer_src", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
