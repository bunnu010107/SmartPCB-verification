"""Plain-assert tests for quality_control/rules.py. Run: python tests/test_rules.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quality_control.rules import build_report, load_config, validate_config  # noqa: E402

CFG = load_config(ROOT / "quality_control" / "config_v1_1024.json")
BOX = [[100, 100], [140, 100], [140, 120], [100, 120]]  # 40 x 20 px rectangle


def det(class_id, conf):
    return {"class_id": class_id, "confidence": conf, "polygon": BOX}


def report(dets):
    return build_report(dets, CFG, "T-1", "x.jpg", [2048, 1500], timestamp="2026-11-15T00:00:00+00:00")


def test_no_detections_is_pass():
    r = report([])
    assert r["board_status"] == "PASS" and r["defects"] == []
    assert sum(r["defect_counts"].values()) == 0


def test_just_below_review_is_pass():
    assert report([det(1, 0.399)])["board_status"] == "PASS"


def test_review_boundary():
    r = report([det(1, 0.40)])
    assert r["board_status"] == "REVIEW" and r["defects"][0]["tier"] == "possible"


def test_hold_boundary():
    r = report([det(2, 0.50)])
    assert r["board_status"] == "HOLD" and r["defects"][0]["tier"] == "confirmed"


def test_mixed_is_hold_and_sorted():
    r = report([det(4, 0.45), det(3, 0.90)])
    assert r["board_status"] == "HOLD"
    assert [d["class"] for d in r["defects"]] == ["short", "spur"]


def test_counts_per_class():
    r = report([det(1, 0.9), det(1, 0.6), det(4, 0.45), det(0, 0.1)])
    assert r["defect_counts"]["mouse_bite"] == 2
    assert r["defect_counts"]["spur"] == 1
    assert r["defect_counts"]["missing_pad"] == 0


def test_geometry():
    d = report([det(0, 0.9)])["defects"][0]
    assert d["center_px"] == [120.0, 110.0] and d["size_px"] == [40.0, 20.0]


def test_report_is_json_serialisable():
    text = json.dumps(report([det(0, 0.9), det(5, 0.42)]))
    assert json.loads(text)["model"]["imgsz"] == 1024


def test_bad_config_rejected():
    bad = dict(CFG, review_conf=0.6, hold_conf=0.5)
    try:
        validate_config(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"ALL {len(tests)} TESTS PASSED")
