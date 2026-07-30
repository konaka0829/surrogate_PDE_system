from __future__ import annotations

from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
HEX_64 = re.compile(r"""["']([0-9a-f]{64})["']""")


def test_every_hard_coded_test_hash_is_classified_in_inventory() -> None:
    actual: Counter[tuple[str, str]] = Counter()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        actual.update(
            (path.name, match.group(1))
            for match in HEX_64.finditer(text)
        )
    mock_hash = (
        "d693c22500c07511a76bfb36"
        "f5b8227616c87692c8f5448be32b3538412ffc99"
    )
    split_hash = (
        "80e69dd30b1caa4acae417297"
        "89c90449c8749292a6ceb85680949656dd503e1"
    )
    assert actual == Counter(
        {
            ("test_random_feature_evaluation.py", mock_hash): 1,
            ("test_validation_data.py", split_hash): 3,
        }
    )
    inventory = (
        ROOT / "docs" / "numerical_hash_inventory.md"
    ).read_text(encoding="utf-8")
    assert mock_hash in inventory
    assert split_hash in inventory
    assert "mock_fixture_split_hash" in inventory
    assert "scientific_split_hash" in inventory
