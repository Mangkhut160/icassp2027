"""Verify a multi-demo balanced dual-moka source feasibility batch.

Scope: partial rollout (env_step=0).  The layout verifier performs the
existing MuJoCo-derived artifact checks; this wrapper adds exact demo/source
binding and rejects duplicate or incomplete state pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.verify_libero_balanced_dual_moka import verify as verify_layout
from tools.verify_libero_balanced_dual_moka_pair_batch import _same_vector


CLASSIFICATION = "partial rollout"
SOURCE_SCHEMA = "odeworld_balanced_dual_moka_multidemo_source_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_digest(manifest: dict[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _regular_dir(path: Path, label: str) -> Path:
    path = Path(path)
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"{label} must be a regular directory: {path}")
    return path.resolve()


def verify_multidemo(
    output_root: Path,
    source_root: Path,
    expected_demos: list[str],
) -> dict[str, Any]:
    output_root = _regular_dir(output_root, "batch root")
    source_root = _regular_dir(source_root, "source root")
    source_manifest = _json(source_root / "source_manifest.json")
    if source_manifest.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError("Unexpected source manifest schema")
    source_records = {str(row.get("demo")): row for row in source_manifest.get("states", [])}
    if list(source_records) != expected_demos:
        raise RuntimeError("Source manifest demo order does not match expected demos")

    pair_dirs = sorted(
        (path for path in output_root.iterdir() if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.name,
    )
    expected_pairs = [f"pair_{index:02d}" for index in range(len(expected_demos))]
    checks: dict[str, bool] = {
        "exact_pair_dirs": [path.name for path in pair_dirs] == expected_pairs,
        "source_manifest_hash": _manifest_digest(source_manifest)
        == str(source_manifest.get("manifest_sha256", "")),
    }
    details: dict[str, Any] = {"pairs": {}}
    seen_states: set[str] = set()
    for index, demo_name in enumerate(expected_demos):
        pair_name = f"pair_{index:02d}"
        pair_dir = output_root / pair_name
        a_dir, b_dir = pair_dir / "layout_a", pair_dir / "layout_b"
        checks[f"{pair_name}_dirs"] = (
            pair_dir.is_dir()
            and not pair_dir.is_symlink()
            and a_dir.is_dir()
            and not a_dir.is_symlink()
            and b_dir.is_dir()
            and not b_dir.is_symlink()
        )
        if not checks[f"{pair_name}_dirs"]:
            continue
        a = _json(a_dir / "layout_summary.json")
        b = _json(b_dir / "layout_summary.json")
        a_scope, b_scope = a.get("scope", {}), b.get("scope", {})
        source_record = source_records[demo_name]
        state_hash = str(source_record.get("recorded_state_sha256", ""))
        seen_states.add(state_hash)
        checks[f"{pair_name}_layout_checks"] = bool(
            verify_layout(a_dir).get("all_checks_passed")
            and verify_layout(b_dir).get("all_checks_passed")
        )
        checks[f"{pair_name}_demo_binding"] = (
            a_scope.get("demo") == demo_name
            and b_scope.get("demo") == demo_name
            and a_scope.get("hdf5_path") == b_scope.get("hdf5_path")
            == source_manifest.get("official_hdf5", {}).get("path")
            and a_scope.get("scene_dir") == b_scope.get("scene_dir")
            == str((source_root / demo_name).resolve())
        )
        checks[f"{pair_name}_source_xml"] = (
            _sha256(source_root / demo_name / "scene.xml")
            == source_record.get("dual_scene_xml_sha256")
        )
        a_selected, b_selected = a.get("selected_candidate", {}), b.get("selected_candidate", {})
        checks[f"{pair_name}_orientations"] = (
            a_scope.get("orientation") == "A" and b_scope.get("orientation") == "B"
        )
        checks[f"{pair_name}_identity_swap"] = (
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
        details["pairs"][pair_name] = {
            "demo": demo_name,
            "state_sha256": state_hash,
            "candidate_a": a_selected.get("candidate_id"),
            "candidate_b": b_selected.get("candidate_id"),
            "layout_a_summary_sha256": _sha256(a_dir / "layout_summary.json"),
            "layout_b_summary_sha256": _sha256(b_dir / "layout_summary.json"),
        }

    checks["unique_state_hashes"] = len(seen_states) == len(expected_demos)
    checks["all_checks_passed"] = all(checks.values())
    result = {
        "classification": CLASSIFICATION,
        "schema": "odeworld_balanced_dual_moka_multidemo_batch_v1",
        "expected_demos": expected_demos,
        "pair_count": len(pair_dirs),
        "checks": checks,
        "details": details,
        "all_checks_passed": checks["all_checks_passed"],
    }
    (output_root / "multidemo_verification_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    if not result["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Multi-demo verification failed: {failed}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--expected-demos", nargs="+", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_multidemo(args.output_root, args.source_root, args.expected_demos),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
