# layout_cli.py
# -*- coding: utf-8 -*-
"""
CLI for universal layout classifier using a single config.yaml.
Commands:
  python layout_cli.py --config config.yaml build
  python layout_cli.py --config config.yaml learn
  python layout_cli.py --config config.yaml validate
  python layout_cli.py --config config.yaml predict
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

from .label_layout_classifier import (
    build_index,
    learn_thresholds,
    validate_open_set,
    predict_batch,
    predict_one,
    list_labels as _list_labels,
)



def read_cfg(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() in (".yml", ".yaml"):
        if yaml is None:
            raise RuntimeError("PyYAML is required for YAML configs. Install with: pip install pyyaml")
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    else:
        return json.loads(p.read_text(encoding="utf-8"))


def save_meta(model_dir: Path, params: dict):
    meta = {
        "version": "layout-1.0.0",
        "params": params,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_labelset(model_dir: Path, labels):
    (model_dir / "labelset.txt").write_text("\n".join(labels), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(prog="layout", description="Layout classifier CLI (config-driven)")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("cmd", choices=["build", "learn", "validate", "predict"])
    args = ap.parse_args()

    cfg = read_cfg(args.config)
    model_dir = Path(cfg.get("model_dir", "model"))
    data = cfg.get("data", {}) or {}
    assets = cfg.get("assets", {}) or {}
    params = cfg.get("params", {}) or {}
    learn_cfg = cfg.get("learn", {}) or {}
    pred_cfg = cfg.get("predict", {}) or {}

    # paths
    train_dir = Path(data.get("train_dir", "")) if data.get("train_dir") else None
    val_dir = Path(data.get("val_dir", "")) if data.get("val_dir") else None
    unknown_dir = Path(data.get("unknown_dir", "")) if data.get("unknown_dir") else None
    avg_dir = Path(assets.get("avg_dir")) if assets.get("avg_dir") else None
    roi_dir = Path(assets.get("roi_dir")) if assets.get("roi_dir") else None

    index_path = model_dir / "index.npz"

    # thresholds: accept both names
    thr1 = model_dir / "thresholds.json"
    thr2 = model_dir / "per_class_thresholds.json"
    thresholds_json = thr1 if thr1.exists() else (thr2 if thr2.exists() else None)

    # defaults aligned with your module
    P = dict(
        crop_left_percent=float(params.get("crop_left_percent", 1.0)),
        alpha=float(params.get("alpha", 0.60)),
        beta=float(params.get("beta", 0.20)),
        gamma=float(params.get("gamma", 0.20)),
        tau=float(params.get("tau", 0.92)),
        margin=float(params.get("margin", 0.015)),
        tau_high=float(params.get("tau_high", 0.98)),
        fast_accept=float(params.get("fast_accept", 0.955)),
        top3_gap=float(params.get("top3_gap", 0.03)),
    )

    if args.cmd == "build":
        if not train_dir:
            sys.exit("train_dir missing in config.")
        # Build index
        build_index(train_dir, avg_dir, index_path, P["crop_left_percent"])
        # Save meta + labelset + version
        save_meta(model_dir, P)
        labs = _list_labels(train_dir)
        write_labelset(model_dir, labs)
        (model_dir / "version.txt").write_text("layout-1.0.0", encoding="utf-8")
        print("[OK] build completed.")

    elif args.cmd == "learn":
        if not val_dir:
            sys.exit("val_dir missing in config.")
        out_thr = thr1  # write as thresholds.json
        learn_thresholds(
            data_dir=val_dir, index_path=index_path, crop_left_percent=P["crop_left_percent"],
            alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"],
            percentile=float(learn_cfg.get("percentile", 0.05)),
            margin_percentile=float(learn_cfg.get("margin_percentile", 0.10)),
            out_json=out_thr
        )
        print("[OK] learn completed.")

    elif args.cmd == "validate":
        if not val_dir:
            sys.exit("val_dir missing in config.")
        validate_open_set(
            data_dir=val_dir, unknown_dir=unknown_dir, index_path=index_path,
            crop_left_percent=P["crop_left_percent"], alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"],
            tau=P["tau"], margin=P["margin"], tau_high=P["tau_high"],
            fast_accept=P["fast_accept"], top3_gap=P["top3_gap"],
            thresholds_json=thresholds_json
        )

    elif args.cmd == "predict":
        inp = pred_cfg.get("input")
        if not inp:
            sys.exit("predict.input missing in config.")
        inp = Path(inp)
        out_csv = Path(pred_cfg.get("out_csv", "results/preds.csv"))
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        if inp.is_file():
            predict_one(
                image_path=inp, index_path=index_path, avg_dir=avg_dir, roi_dir=roi_dir,
                crop_left_percent=P["crop_left_percent"], alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"],
                tau=P["tau"], margin=P["margin"], tau_high=P["tau_high"],
                fast_accept=P["fast_accept"], top3_gap=P["top3_gap"],
                thresholds_json=thresholds_json
            )
        else:
            predict_batch(
                in_dir=inp, index_path=index_path, avg_dir=avg_dir, roi_dir=roi_dir, out_csv=out_csv,
                crop_left_percent=P["crop_left_percent"], alpha=P["alpha"], beta=P["beta"], gamma=P["gamma"],
                tau=P["tau"], margin=P["margin"], tau_high=P["tau_high"],
                fast_accept=P["fast_accept"], top3_gap=P["top3_gap"],
                thresholds_json=thresholds_json
            )
        print("[OK] predict completed.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
