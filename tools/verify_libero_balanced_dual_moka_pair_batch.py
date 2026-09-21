#!/usr/bin/env python3
"""Verify paired A/B balanced dual-moka input state artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.verify_libero_balanced_dual_moka import verify

CLASSIFICATION = "partial rollout"


def _same_vector(first: Any, second: Any, tolerance: float = 1e-9) -> bool:
    """Compare serialized coordinates without rejecting harmless float noise."""
    if not isinstance(first, list) or not isinstance(second, list) or len(first) != len(second):
        return False
    try:
        return all(abs(float(left) - float(right)) <= tolerance for left, right in zip(first, second))
    except (TypeError, ValueError):
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected summary object: {path}")
    return value


def verify_pair_batch(output_root: Path, expected_pairs: int) -> dict[str, Any]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise RuntimeError(f"Expected regular pilot root: {output_root}")
    pair_dirs = sorted(
        path for path in output_root.iterdir() if path.is_dir() and not path.is_symlink()
    )
    expected_names = [f"pair_{index:02d}" for index in range(expected_pairs)]
    checks: dict[str, bool] = {
        "exact_pair_dirs": [path.name for path in pair_dirs] == expected_names,
    }
    details: dict[str, Any] = {"pairs": {}}
    for pair_dir in pair_dirs:
        if pair_dir.name not in expected_names:
            continue
        a_dir, b_dir = pair_dir / "layout_a", pair_dir / "layout_b"
        checks[f"{pair_dir.name}_layout_dirs"] = (
            a_dir.is_dir() and not a_dir.is_symlink() and b_dir.is_dir() and not b_dir.is_symlink()
        )
        if not checks[f"{pair_dir.name}_layout_dirs"]:
            continue
        a_result, b_result = verify(a_dir), verify(b_dir)
        a, b = _summary(a_dir / "layout_summary.json"), _summary(b_dir / "layout_summary.json")
        a_scope, b_scope = a.get("scope", {}), b.get("scope", {})
        a_selected, b_selected = a.get("selected_candidate", {}), b.get("selected_candidate", {})
        checks[f"{pair_dir.name}_checks"] = bool(
            a_result.get("all_checks_passed") and b_result.get("all_checks_passed")
        )
        checks[f"{pair_dir.name}_pair_index"] = (
            a_scope.get("pair_index") == b_scope.get("pair_index") == int(pair_dir.name[-2:])
        )
        checks[f"{pair_dir.name}_orientations"] = (
            a_scope.get("orientation") == "A" and b_scope.get("orientation") == "B"
        )
        checks[f"{pair_dir.name}_identity_swap"] = (
            a_selected.get("white_xy_m") == b_selected.get("red_xy_m")
            and a_selected.get("red_xy_m") == b_selected.get("white_xy_m")
            and _same_vector(
                a.get("settled_layout", {}).get("white_position_xyz_m"),
                b.get("settled_layout", {}).get("red_position_xyz_m"),
            )
            and _same_vector(
                a.get("settled_layout", {}).get("red_position_xyz_m"),
                b.get("settled_layout", {}).get("white_position_xyz_m"),
            )
        )
        details["pairs"][pair_dir.name] = {
            "layout_a_summary_sha256": _sha256(a_dir / "layout_summary.json"),
            "layout_b_summary_sha256": _sha256(b_dir / "layout_summary.json"),
            "layout_a_verification_sha256": _sha256(a_dir / "verification_summary.json"),
            "layout_b_verification_sha256": _sha256(b_dir / "verification_summary.json"),
            "selected_candidate": a_selected.get("candidate_id"),
        }
    checks["all_checks_passed"] = all(checks.values())
    result = {
        "classification": CLASSIFICATION,
        "schema": "odeworld_balanced_dual_moka_pair_batch_v1",
        "expected_pairs": expected_pairs,
        "pair_count": len(pair_dirs),
        "checks": checks,
        "details": details,
        "all_checks_passed": checks["all_checks_passed"],
    }
    (output_root / "batch_verification_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not result["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Balanced pair batch verification failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--expected-pairs", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(verify_pair_batch(args.output_root, args.expected_pairs), indent=2), flush=True)


if __name__ == "__main__":
    main()
