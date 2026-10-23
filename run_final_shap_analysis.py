#!/usr/bin/env python3
"""Final SHAP analysis for the selected CAMEO XGBoost radiation models.

This script explains the already-fitted final GHI, DHI, and BNI models.  It
reuses the deterministic station/date splits from the grid search, builds the
SHAP background only from training + validation rows, and computes dataset-
level results on the untouched test rows.  It also recreates the two example
waterfalls used by the manuscript; their original observations are in the
training partition and are labelled as such in the saved metadata.

Outputs include reusable SHAP data, tabular feature/category summaries, and
publication figures in both PDF and PNG formats.


The MIT License

Copyright 2026 Antti Lipponen / Finnish Meteorological Institute

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cameo_final_shap_mpl_cache")
)

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, PathCollection
from matplotlib.patches import Circle, Polygon, Rectangle, RegularPolygon
import numpy as np
import pandas as pd
import shap
import xgboost as xgb


CASES = ("GHI", "DHI", "BNI")
SHAP_XRANGES = {"GHI": (-500.0, 500.0), "DHI": (-300.0, 300.0), "BNI": (-600.0, 600.0)}
PLOT_EXCLUDED_FEATURES = ("AODSO",)
CATEGORY_FEATURES = {
    "Aerosol": ["AODNI", "AODAM", "AODBC", "AODOR", "AODSU", "AODSS", "AODDU"],
    "Cloud": ["CloudCoverage", "CloudOpticalDepth", "CloudType"],
    "Surface": ["fgeo", "fiso", "fvol", "albedo"],
    "Gas": ["tcwv", "tco3"],
}
CATEGORY_COLORS = {
    "Aerosol": "#FFA500",
    "Cloud": "#1F77FF",
    "Surface": "#228B22",
    "Gas": "#800080",
    "Others": "#808080",
}
EXAMPLES = {
    "2611": {"station": "BSRN_PAL", "timestamp": "2021-11-14 09:26:00"},
    "5734": {"station": "BSRN_CAB", "timestamp": "2021-08-05 17:11:00"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=json_default)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute final-model SHAP results using a train+validation background "
            "and the held-out station/date test partition."
        )
    )
    default_root = Path(__file__).resolve().parent
    parser.add_argument("--data-root", type=Path, default=default_root)
    parser.add_argument(
        "--search-results",
        type=Path,
        default=None,
        help="Default: DATA_ROOT/grid_search_results",
    )
    parser.add_argument(
        "--final-results",
        type=Path,
        default=None,
        help="Folder containing CASE/final_model.json (default: DATA_ROOT/final_model_results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: DATA_ROOT/final_shap_results",
    )
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument(
        "--background-samples",
        type=int,
        default=100,
        help="Deterministic train+validation background size for interventional TreeSHAP.",
    )
    parser.add_argument(
        "--max-explain-samples",
        type=int,
        default=None,
        help="Optional deterministic test subset for a quick verification run; default explains all test rows.",
    )
    parser.add_argument(
        "--beeswarm-samples",
        type=int,
        default=20000,
        help="Maximum deterministic points rendered in a beeswarm (all rows still contribute to statistics).",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--nthread", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--force", action="store_true", help="Replace completed case outputs.")
    parser.add_argument("--allow-sleep", action="store_true")
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.search_results = (
        args.search_results.expanduser().resolve()
        if args.search_results is not None
        else args.data_root / "grid_search_results"
    )
    args.final_results = (
        args.final_results.expanduser().resolve()
        if args.final_results is not None
        else args.data_root / "final_model_results"
    )
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.data_root / "final_shap_results"
    )
    if args.background_samples <= 0 or args.beeswarm_samples <= 0 or args.nthread <= 0:
        parser.error("sample counts and --nthread must be positive")
    if args.max_explain_samples is not None and args.max_explain_samples <= 0:
        parser.error("--max-explain-samples must be positive")
    return args


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cameo_final_shap")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    # Append so a resumed run retains the earlier case-computation history.
    file_handler = logging.FileHandler(output_dir / "final_shap.log", mode="a")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def prevent_macos_sleep(logger: logging.Logger, allow_sleep: bool) -> subprocess.Popen | None:
    if allow_sleep or platform.system() != "Darwin" or shutil.which("caffeinate") is None:
        return None
    try:
        process = subprocess.Popen(
            ["caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Idle system sleep is disabled until the SHAP run completes.")
        return process
    except OSError as exc:
        logger.warning("Could not start caffeinate; the Mac may sleep: %s", exc)
        return None


def load_grid_module(data_root: Path):
    path = data_root / "run_xgboost_grid_search.py"
    if not path.is_file():
        raise FileNotFoundError(f"Required companion script not found: {path}")
    specification = importlib.util.spec_from_file_location("cameo_grid_search", path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def deterministic_rows(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(frame):
        return frame.copy()
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(frame), size=n, replace=False))
    return frame.iloc[positions].copy()


def build_categories(features: list[str]) -> dict[str, list[str]]:
    categories = {
        name: [feature for feature in members if feature in features]
        for name, members in CATEGORY_FEATURES.items()
    }
    assigned = {feature for members in categories.values() for feature in members}
    categories["Others"] = [feature for feature in features if feature not in assigned]
    return categories


def save_figure(fig: plt.Figure, base_path: Path, *, tight: bool = False) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(base_path.with_suffix(".pdf"), **kwargs)
    fig.savefig(base_path.with_suffix(".png"), dpi=220, **kwargs)
    plt.close(fig)


def plot_explanation_without_aodso(explanation: shap.Explanation) -> shap.Explanation:
    names = list(explanation.feature_names)
    keep = [index for index, name in enumerate(names) if name not in PLOT_EXCLUDED_FEATURES]
    return shap.Explanation(
        values=np.asarray(explanation.values)[:, keep],
        base_values=np.asarray(explanation.base_values),
        data=np.asarray(explanation.data)[:, keep],
        feature_names=[names[index] for index in keep],
    )


def add_feature_category_symbol(fig: plt.Figure, y_figure: float, category: str) -> None:
    color = CATEGORY_COLORS[category]
    x = 0.025
    if category == "Aerosol":
        artist = Circle((x, y_figure), 0.0105, color=color, transform=fig.transFigure)
    elif category == "Cloud":
        artist = Rectangle((x - 0.010, y_figure - 0.009), 0.020, 0.018, color=color, transform=fig.transFigure)
    elif category == "Surface":
        artist = Polygon(
            [[x, y_figure + 0.010], [x - 0.012, y_figure - 0.010], [x + 0.012, y_figure - 0.010]],
            color=color,
            transform=fig.transFigure,
        )
    elif category == "Gas":
        artist = Polygon(
            [[x, y_figure + 0.011], [x - 0.010, y_figure], [x, y_figure - 0.011], [x + 0.010, y_figure]],
            color=color,
            transform=fig.transFigure,
        )
    else:
        artist = RegularPolygon(
            (x, y_figure), numVertices=5, radius=0.0115, orientation=np.pi / 3,
            color=color, transform=fig.transFigure,
        )
    artist.set_clip_on(False)
    artist.set_zorder(100)
    fig.add_artist(artist)


def feature_category(feature: str) -> str:
    for category, members in CATEGORY_FEATURES.items():
        if feature in members:
            return category
    return "Others"


def make_beeswarm(
    case: str,
    explanation: shap.Explanation,
    output_dir: Path,
    seed: int,
    maximum_points: int,
) -> None:
    plotted = plot_explanation_without_aodso(explanation)
    values = np.asarray(plotted.values)
    data = np.asarray(plotted.data).copy()
    names = list(plotted.feature_names)
    # CloudType uses 5--8 in the source data; remapping only affects color display.
    if "CloudType" in names:
        cloud_index = names.index("CloudType")
        remap = {5.0: 1.0, 6.0: 2.0, 7.0: 3.0, 8.0: 4.0}
        for old, new in remap.items():
            data[data[:, cloud_index] == old, cloud_index] = new
    mean_abs = np.mean(np.abs(values), axis=0)
    labels = [f"{name}: {mean_abs[index]:.1f} W/m²" for index, name in enumerate(names)]
    explanation_for_plot = shap.Explanation(
        values=values,
        base_values=np.asarray(plotted.base_values),
        data=data,
        feature_names=labels,
    )
    if len(values) > maximum_points:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(len(values), size=maximum_points, replace=False))
        explanation_for_plot = explanation_for_plot[positions]

    order = np.argsort(-mean_abs)
    fig = plt.figure(figsize=(8.0, 9.1))
    # Leave enough room for the manuscript's metric-bearing feature labels.
    ax = fig.add_axes([0.46, 0.10, 0.415, 0.87])
    shap.plots.beeswarm(
        explanation_for_plot,
        max_display=len(names),
        order=order,
        s=10,
        alpha=0.20,
        show=False,
        plot_size=None,
        ax=ax,
    )
    xmin, xmax = SHAP_XRANGES[case]
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(
        rf"SHAP value: impact on ${case}_{{obs}} - {case}_{{CAMS}}$ (W/m²)",
        fontsize=13,
        labelpad=10,
    )

    # Reproduce the manuscript's faint red/blue high/low-value dominance bands.
    positive_rectangles: list[Rectangle] = []
    negative_rectangles: list[Rectangle] = []
    bin_edges = np.linspace(xmin, xmax, 51)
    for display_row, feature_index in enumerate(order):
        shap_feature = values[:, feature_index]
        feature_data = data[:, feature_index]
        finite = np.isfinite(feature_data)
        if not finite.any():
            continue
        q05, q95 = np.nanpercentile(feature_data, [5, 95])
        if q05 == q95:
            q01, q99 = np.nanpercentile(feature_data, [1, 99])
            midpoint = np.mean([q01, q99])
        else:
            midpoint = np.mean([q05, q95])
        bin_index = np.digitize(shap_feature, bin_edges) - 1
        y_bottom = len(names) - display_row - 1.5
        for this_bin in range(50):
            selected = (bin_index == this_bin) & finite
            if not selected.any():
                continue
            centered = feature_data[selected] - midpoint
            rectangle = Rectangle(
                (bin_edges[this_bin], y_bottom),
                bin_edges[this_bin + 1] - bin_edges[this_bin],
                1.0,
            )
            if np.mean(centered > 0) >= 0.90:
                positive_rectangles.append(rectangle)
            elif np.mean(centered < 0) >= 0.90:
                negative_rectangles.append(rectangle)
    if positive_rectangles:
        ax.add_collection(PatchCollection(positive_rectangles, facecolor="#FF1F1F", alpha=0.10, edgecolor="none", zorder=-10))
    if negative_rectangles:
        ax.add_collection(PatchCollection(negative_rectangles, facecolor="#1F77FF", alpha=0.10, edgecolor="none", zorder=-10))
    for artist in ax.get_children():
        if isinstance(artist, PathCollection):
            artist.set_rasterized(True)
    fig.canvas.draw()
    ordered_names = [names[index] for index in order]
    for row, feature in enumerate(ordered_names):
        y_data = len(names) - row - 1
        y_display = ax.transData.transform((0, y_data))[1]
        y_figure = fig.transFigure.inverted().transform((0, y_display))[1]
        add_feature_category_symbol(fig, y_figure, feature_category(feature))
    save_figure(fig, output_dir / "figures" / f"shap_beeswarm_{case}")


def category_sza_table(
    case: str,
    explanation: shap.Explanation,
    observed: np.ndarray,
) -> pd.DataFrame:
    plotted = plot_explanation_without_aodso(explanation)
    values = pd.DataFrame(np.asarray(plotted.values), columns=plotted.feature_names)
    data = pd.DataFrame(np.asarray(plotted.data), columns=plotted.feature_names)
    categories = build_categories(list(plotted.feature_names))
    bins = np.arange(0.0, 82.5, 2.5)
    bin_index = pd.cut(data["sza"], bins=bins, right=False, labels=False, include_lowest=True)
    rows: list[dict[str, Any]] = []
    observed = np.asarray(observed, dtype=np.float64)
    for index in range(len(bins) - 1):
        selected = np.asarray(bin_index == index)
        count = int(np.sum(selected))
        for category, members in categories.items():
            signed_group = values[members].sum(axis=1).to_numpy(dtype=np.float64)
            normalized_abs = np.abs(signed_group / observed)
            rows.append(
                {
                    "case": case,
                    "category": category,
                    "sza_left_degrees": bins[index],
                    "sza_right_degrees": bins[index + 1],
                    "sza_center_degrees": 0.5 * (bins[index] + bins[index + 1]),
                    "mean_normalized_absolute_shap": (
                        float(np.mean(normalized_abs[selected])) if count else float("nan")
                    ),
                    "sample_count": count,
                }
            )
    return pd.DataFrame(rows)


def make_sza_plot(case: str, table: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for category in ("Aerosol", "Cloud", "Surface", "Gas", "Others"):
        part = table.loc[table["category"] == category].sort_values("sza_left_degrees")
        x = np.r_[part["sza_left_degrees"].to_numpy(), part["sza_right_degrees"].to_numpy()[-1]]
        y = np.r_[part["mean_normalized_absolute_shap"].to_numpy(), part["mean_normalized_absolute_shap"].to_numpy()[-1]]
        ax.step(x, y, where="post", linewidth=2.0, label=category, color=CATEGORY_COLORS[category])
    counts = table.loc[table["category"] == "Others"].sort_values("sza_center_degrees")
    for row in counts.itertuples(index=False):
        ax.text(
            row.sza_center_degrees,
            1.01,
            str(int(row.sample_count)),
            fontsize=6,
            ha="center",
            va="bottom",
            rotation=90,
            clip_on=False,
            transform=ax.get_xaxis_transform(),
        )
    ax.set_xlim(0, 80)
    ax.set_xlabel("Solar zenith angle (°)", fontsize=12)
    ax.set_ylabel("Normalized mean absolute SHAP value", fontsize=12)
    ax.grid(True)
    ax.legend()
    fig.suptitle(case, fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / f"shap_sza_relative_steps_{case}")


def save_waterfall(
    case: str,
    example_id: str,
    explanation: shap.Explanation,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    shap.plots.waterfall(explanation[0], max_display=10, show=False)
    ax.set_title(case, fontsize=15)
    fig.set_size_inches(10.0, 5.0)
    fig.tight_layout()
    save_figure(fig, output_dir / "figures" / f"shap_waterfall_{case}_example_{example_id}")


def summarize_features(case: str, explanation: shap.Explanation) -> pd.DataFrame:
    values = np.asarray(explanation.values)
    signed = np.mean(values, axis=0)
    absolute = np.mean(np.abs(values), axis=0)
    frame = pd.DataFrame(
        {
            "case": case,
            "feature": explanation.feature_names,
            "category": [feature_category(name) for name in explanation.feature_names],
            "mean_absolute_shap_W_m2": absolute,
            "mean_signed_shap_W_m2": signed,
        }
    )
    frame["rank"] = frame["mean_absolute_shap_W_m2"].rank(method="first", ascending=False).astype(int)
    return frame.sort_values("rank").reset_index(drop=True)


def summarize_categories(case: str, explanation: shap.Explanation) -> pd.DataFrame:
    names = list(explanation.feature_names)
    values = pd.DataFrame(np.asarray(explanation.values), columns=names)
    rows = []
    for category, members in build_categories(names).items():
        group = values[members].sum(axis=1).to_numpy(dtype=np.float64)
        rows.append(
            {
                "case": case,
                "category": category,
                "mean_absolute_group_shap_W_m2": float(np.mean(np.abs(group))),
                "mean_signed_group_shap_W_m2": float(np.mean(group)),
                "features": ", ".join(members),
            }
        )
    result = pd.DataFrame(rows).sort_values("mean_absolute_group_shap_W_m2", ascending=False)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def load_booster(path: Path, nthread: int) -> xgb.Booster:
    if not path.is_file():
        raise FileNotFoundError(f"Final model not found: {path}")
    booster = xgb.Booster({"nthread": nthread})
    booster.load_model(path)
    return booster


def process_case(
    case: str,
    args: argparse.Namespace,
    grid_module,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    case_dir = args.output_dir / case
    completion_path = case_dir / "metadata.json"
    if completion_path.exists() and not args.force:
        logger.info("%s: completed output exists; use --force to replace it", case)
        return (
            pd.read_csv(case_dir / "mean_abs_shap.csv"),
            pd.read_csv(case_dir / "category_importance.csv"),
            pd.read_csv(case_dir / "case_examples.csv").to_dict("records"),
        )
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    logger.info("%s: loading data, saved split, and final model", case)
    frame = grid_module.load_case(args.data_root, case, logger)
    row_splits = grid_module.make_or_load_splits(
        frame,
        args.search_results / case / "station_date_splits.csv",
        args.seed,
        0.70,
        0.15,
        logger,
    )
    features = list(grid_module.FEATURES)
    X_all = frame[features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    target_all = (frame[f"{case}_obs"] - frame[case]).to_numpy(dtype=np.float32)
    trainval_mask = row_splits.isin(["train", "validation"]).to_numpy()
    test_mask = (row_splits == "test").to_numpy()
    X_trainval = X_all.loc[trainval_mask].reset_index(drop=True)
    X_test_full = X_all.loc[test_mask].reset_index(drop=True)
    test_frame_full = frame.loc[test_mask].reset_index(drop=True)
    y_trainval = target_all[trainval_mask]
    y_test_full = target_all[test_mask]
    if args.max_explain_samples is not None:
        selected = deterministic_rows(
            pd.DataFrame({"position": np.arange(len(X_test_full))}),
            min(args.max_explain_samples, len(X_test_full)),
            args.seed + CASES.index(case) * 1000 + 11,
        )["position"].to_numpy(dtype=int)
        X_test = X_test_full.iloc[selected].reset_index(drop=True)
        test_frame = test_frame_full.iloc[selected].reset_index(drop=True)
        y_test = y_test_full[selected]
    else:
        X_test = X_test_full
        test_frame = test_frame_full
        y_test = y_test_full
    background = deterministic_rows(
        X_trainval,
        min(args.background_samples, len(X_trainval)),
        args.seed + CASES.index(case) * 1000 + 1,
    ).reset_index(drop=True)
    booster = load_booster(args.final_results / case / "final_model.json", args.nthread)
    logger.info(
        "%s: constructing interventional TreeSHAP with %d train+validation background rows",
        case,
        len(background),
    )
    explainer = shap.Explainer(
        booster,
        background,
        algorithm="tree",
        feature_names=features,
    )
    logger.info("%s: explaining %s held-out test rows", case, f"{len(X_test):,}")
    shap_started = time.perf_counter()
    explanation = explainer(X_test, check_additivity=True)
    shap_seconds = time.perf_counter() - shap_started
    test_prediction = booster.predict(xgb.DMatrix(X_test, feature_names=features, nthread=args.nthread))
    reconstructed = np.asarray(explanation.base_values) + np.sum(np.asarray(explanation.values), axis=1)
    additivity_error = np.abs(reconstructed - test_prediction)
    max_additivity_error = float(np.max(additivity_error))
    # Float32 XGBoost predictions and float64 SHAP reconstruction can differ by
    # a few thousandths of a W/m² for large ensembles.  This tolerance is still
    # far below measurement or reporting precision and supplements SHAP's own
    # successful additivity check above.
    if max_additivity_error > 1e-2:
        raise RuntimeError(f"{case}: SHAP additivity check failed ({max_additivity_error:g})")
    logger.info(
        "%s: SHAP completed in %.1f s; maximum additivity error %.3g W/m²",
        case,
        shap_seconds,
        max_additivity_error,
    )

    feature_table = summarize_features(case, explanation)
    category_table = summarize_categories(case, explanation)
    sza_table = category_sza_table(case, explanation, test_frame[f"{case}_obs"].to_numpy())
    atomic_csv(case_dir / "mean_abs_shap.csv", feature_table)
    atomic_csv(case_dir / "category_importance.csv", category_table)
    atomic_csv(case_dir / "shap_sza_binned.csv", sza_table)
    make_beeswarm(case, explanation, args.output_dir, args.seed + CASES.index(case), args.beeswarm_samples)
    make_sza_plot(case, sza_table, args.output_dir)

    example_rows: list[dict[str, Any]] = []
    for example_id, selector in EXAMPLES.items():
        timestamp = pd.Timestamp(selector["timestamp"])
        selected = (frame["station"] == selector["station"]) & (frame["ObservationPeriod"] == timestamp)
        if int(selected.sum()) != 1:
            raise RuntimeError(
                f"{case}: expected one row for example {example_id}, found {int(selected.sum())}"
            )
        source_index = int(np.flatnonzero(selected.to_numpy())[0])
        partition = str(row_splits.iloc[source_index])
        example_X = X_all.iloc[[source_index]]
        example_explanation = explainer(example_X, check_additivity=True)
        save_waterfall(case, example_id, example_explanation, args.output_dir)
        row = frame.iloc[source_index]
        predicted_correction = float(booster.predict(xgb.DMatrix(example_X, feature_names=features))[0])
        record: dict[str, Any] = {
            "case": case,
            "example_id": example_id,
            "station": selector["station"],
            "timestamp_UTC": timestamp.isoformat(),
            "partition": partition,
            "observed_W_m2": float(row[f"{case}_obs"]),
            "CAMS_W_m2": float(row[case]),
            "true_correction_W_m2": float(row[f"{case}_obs"] - row[case]),
            "predicted_correction_W_m2": predicted_correction,
            "corrected_CAMS_W_m2": float(row[case] + predicted_correction),
        }
        record.update({feature: float(row[feature]) for feature in features})
        example_rows.append(record)
        logger.warning(
            "%s example %s (%s, %s UTC) belongs to the %s partition; it is retained only to match the manuscript example.",
            case,
            example_id,
            selector["station"],
            timestamp,
            partition,
        )
    atomic_csv(case_dir / "case_examples.csv", pd.DataFrame(example_rows))

    np.savez_compressed(
        case_dir / "shap_values.npz",
        values=np.asarray(explanation.values, dtype=np.float32),
        base_values=np.asarray(explanation.base_values, dtype=np.float32),
        data=np.asarray(explanation.data, dtype=np.float32),
        feature_names=np.asarray(features),
    )
    train_prediction = booster.predict(xgb.DMatrix(X_trainval, feature_names=features, nthread=args.nthread))
    compatibility_payload = {
        "df": test_frame,
        "explainer": explainer,
        "shap_values": explanation,
        "X_train": X_trainval,
        "X_test": X_test,
        "y_train": y_trainval,
        "y_test": y_test,
        "y_train_pred": train_prediction,
        "y_test_pred": test_prediction,
        "X_background": background,
        "partition_metadata": {
            "background_source": ["train", "validation"],
            "explained_partition": "test",
            "test_rows_available": len(X_test_full),
            "test_rows_explained": len(X_test),
            "split_unit": "station-date",
            "plot_excluded_features": list(PLOT_EXCLUDED_FEATURES),
        },
    }
    logger.info("%s: saving reusable explainerData.joblib", case)
    temporary_joblib = case_dir / "explainerData.joblib.tmp"
    joblib.dump(compatibility_payload, temporary_joblib, compress=("bz2", 3))
    os.replace(temporary_joblib, case_dir / "explainerData.joblib")

    final_metrics_path = args.final_results / case / "final_metrics.json"
    final_metadata = json.loads(final_metrics_path.read_text(encoding="utf-8")) if final_metrics_path.is_file() else {}
    metadata = {
        "case": case,
        "created_at": utc_now(),
        "method": "Interventional TreeSHAP; train+validation background; held-out station-date test explanations.",
        "model_path": args.final_results / case / "final_model.json",
        "candidate_id": final_metadata.get("candidate_id"),
        "num_boost_rounds": final_metadata.get("num_boost_rounds"),
        "feature_names": features,
        "plot_excluded_features": list(PLOT_EXCLUDED_FEATURES),
        "training_validation_rows": len(X_trainval),
        "test_rows_available": len(X_test_full),
        "test_rows_explained": len(X_test),
        "background_rows": len(background),
        "background_seed": args.seed + CASES.index(case) * 1000 + 1,
        "explainer": "shap.Explainer(..., algorithm='tree')",
        "feature_perturbation": getattr(explainer, "feature_perturbation", "interventional"),
        "shap_seconds": shap_seconds,
        "total_case_seconds": time.perf_counter() - started,
        "maximum_additivity_error_W_m2": max_additivity_error,
        "examples": example_rows,
    }
    atomic_json(completion_path, metadata)
    return feature_table, category_table, example_rows


def write_summary(
    output_dir: Path,
    feature_tables: list[pd.DataFrame],
    category_tables: list[pd.DataFrame],
    examples: list[dict[str, Any]],
) -> None:
    all_features = pd.concat(feature_tables, ignore_index=True)
    all_categories = pd.concat(category_tables, ignore_index=True)
    atomic_csv(output_dir / "overall_feature_importance.csv", all_features)
    atomic_csv(output_dir / "overall_category_importance.csv", all_categories)
    atomic_csv(output_dir / "all_case_examples.csv", pd.DataFrame(examples))
    lines = [
        "Final XGBoost SHAP analysis",
        "=" * 31,
        "",
        "Models: frozen final grid-search models refitted on training + validation data.",
        "Dataset-level explanations: held-out station/date test observations only.",
        "SHAP background: deterministic sample from training + validation observations only.",
        "AODSO: retained in models and saved SHAP data, excluded only from manuscript figures.",
        "",
    ]
    for case in CASES:
        part = all_features.loc[all_features["case"] == case].sort_values("rank").head(5)
        if part.empty:
            continue
        lines.append(f"{case} leading features by mean absolute SHAP:")
        for row in part.itertuples(index=False):
            lines.append(f"  {int(row.rank)}. {row.feature}: {row.mean_absolute_shap_W_m2:.3f} W/m²")
        category = all_categories.loc[all_categories["case"] == case].sort_values("rank")
        lines.append("  Group ranking: " + " > ".join(category["category"].tolist()))
        lines.append("")
    lines.extend(
        [
            "Manuscript examples:",
            "  Examples 2611 and 5734 are the two single-sample cases of manuscript Sect. 3.4.",
            "  Check the 'partition' column of case_examples.csv: they are not necessarily test-partition observations.",
            "",
        ]
    )
    (output_dir / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.output_dir)
    grid_module = load_grid_module(args.data_root)
    logger.info("Data root: %s", args.data_root)
    logger.info("Final models: %s", args.final_results)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Python: %s", sys.executable)
    logger.info("SHAP %s; XGBoost %s; CPU threads %d", shap.__version__, xgb.__version__, args.nthread)
    sleep_process = prevent_macos_sleep(logger, args.allow_sleep)
    started = time.perf_counter()
    feature_tables: list[pd.DataFrame] = []
    category_tables: list[pd.DataFrame] = []
    example_rows: list[dict[str, Any]] = []
    try:
        for case in args.cases:
            feature_table, category_table, examples = process_case(case, args, grid_module, logger)
            feature_tables.append(feature_table)
            category_tables.append(category_table)
            example_rows.extend(examples)
            write_summary(args.output_dir, feature_tables, category_tables, example_rows)
        atomic_json(
            args.output_dir / "run_manifest.json",
            {
                "created_at": utc_now(),
                "command": sys.argv,
                "data_root": args.data_root,
                "search_results": args.search_results,
                "final_results": args.final_results,
                "output_dir": args.output_dir,
                "cases": args.cases,
                "seed": args.seed,
                "nthread": args.nthread,
                "background_samples": args.background_samples,
                "max_explain_samples": args.max_explain_samples,
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "shap_version": shap.__version__,
                "xgboost_version": xgb.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        logger.info("All requested SHAP cases completed in %.1f seconds", time.perf_counter() - started)
        return 0
    finally:
        if sleep_process is not None and sleep_process.poll() is None:
            sleep_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
