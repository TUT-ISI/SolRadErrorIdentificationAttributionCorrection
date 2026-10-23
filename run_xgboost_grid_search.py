#!/usr/bin/env python3
"""Resumable XGBoost grid search for GHI, DHI, and BNI error correction.

The model target is observation - CAMS.  Station/date groups are assigned to
70% training, 15% validation, and 15% test partitions within each station.
Validation RMSE selects hyperparameters and controls early stopping.  The test
partition is evaluated only for the selected model.

The default search is designed for an overnight CPU run on a Mac. Results
are saved after every fit, so rerunning the same command resumes the search.


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
import hashlib
import itertools
import json
import logging
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Keep Matplotlib's cache out of a potentially unwritable home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cameo_xgb_mpl_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import pandas as pd
import xgboost as xgb


CASES = ("GHI", "DHI", "BNI")
FEATURES = [
    "sza",
    "summerWinterSplit",
    "tco3",
    "tcwv",
    "AODBC",
    "AODDU",
    "AODSS",
    "AODOR",
    "AODSU",
    "AODNI",
    "AODAM",
    "AODSO",
    "Snow_probability",
    "fiso",
    "fvol",
    "fgeo",
    "albedo",
    "CloudOpticalDepth",
    "CloudCoverage",
    "CloudType",
]

# Stations passing the ground-observation quality classification (identical
# for GHI, DHI, and BNI).  Stations absent from the data are ignored.
ACCEPTED_STATIONS = [
    "BSRN_BUD",
    "BSRN_CAB",
    "BSRN_CNR",
    "BSRN_FLO",
    "BSRN_GOB",
    "BSRN_GVN",
    "BSRN_INO",
    "BSRN_IZA",
    "BSRN_LIN",
    "BSRN_NYA",
    "BSRN_PAL",
    "BSRN_PAY",
    "BSRN_RUN",
    "BSRN_SON",
    "BSRN_SYO",
    "SAURAN_CSIR",
    "SAURAN_CUT",
    "SAURAN_NMU",
    "SAURAN_SUN",
    "SAURAN_UPR",
    "enerMENA_ERF",
    "enerMENA_JOR",
    "enerMENA_TN",
    "enerMENA_ZAG",
]

PLOT_MAX = {"GHI": 1500.0, "DHI": 800.0, "BNI": 1500.0}
SPLIT_NAMES = ("train", "validation", "test")


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


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cameo_xgboost_grid")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "grid_search.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
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
        logger.info("Idle system sleep is disabled until this process exits.")
        return process
    except OSError as exc:
        logger.warning("Could not start caffeinate; the Mac may sleep: %s", exc)
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable CPU XGBoost grid search for radiation error correction."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Folder containing GHIdata, DHIdata, and BNIdata (default: script folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Results folder (default: DATA_ROOT/grid_search_results).",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=list(CASES),
        help="Radiation cases to run (default: GHI DHI BNI).",
    )
    parser.add_argument(
        "--max-hours",
        type=float,
        default=11.0,
        help="Overall wall-clock budget in hours (default: 11).",
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="XGBoost CPU threads (default: all logical CPUs except one).",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--max-boost-rounds",
        type=int,
        default=3000,
        help="Maximum trees per candidate; early stopping usually ends sooner.",
    )
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument(
        "--max-candidates-per-case",
        type=int,
        default=None,
        help="Optional cap, mainly useful for a short smoke test.",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry combinations previously recorded with an error.",
    )
    parser.add_argument(
        "--allow-sleep",
        action="store_true",
        help="Do not keep macOS awake while the script runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show the search size without fitting models.",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.data_root / "grid_search_results"
    )
    if not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between 0 and 1")
    if args.train_fraction + args.validation_fraction >= 1.0:
        parser.error("training + validation fractions must be less than 1")
    if args.max_hours <= 0 or args.nthread <= 0:
        parser.error("--max-hours and --nthread must be positive")
    if args.max_boost_rounds <= 0 or args.early_stopping_rounds <= 0:
        parser.error("boosting and early-stopping rounds must be positive")
    if args.max_candidates_per_case is not None and args.max_candidates_per_case <= 0:
        parser.error("--max-candidates-per-case must be positive")
    return args


def validate_input_files(data_root: Path, cases: Iterable[str]) -> None:
    missing = []
    for case in cases:
        for filename in ("CAMS_dataset.h5", "obs_dataset.h5"):
            path = data_root / f"{case}data" / filename
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing required input file(s):\n" + "\n".join(missing))


def candidate_grid(seed: int, case: str) -> list[dict[str, Any]]:
    """Return a deterministic shuffled Cartesian grid (1,350 candidates)."""
    regularization_profiles = [
        {"reg_alpha": 0.0, "reg_lambda": 1.0, "gamma": 0.0},
        {"reg_alpha": 0.1, "reg_lambda": 5.0, "gamma": 0.0},
        {"reg_alpha": 1.0, "reg_lambda": 10.0, "gamma": 0.1},
    ]
    combinations = []
    for values in itertools.product(
        (0.025, 0.05, 0.10, 0.20, 0.40),  # learning_rate
        (4, 6, 8, 10, 15),  # max_depth
        (1, 5, 15),  # min_child_weight
        (0.70, 0.90),  # subsample
        (0.60, 0.85, 1.00),  # colsample_bytree
        regularization_profiles,
    ):
        learning_rate, max_depth, min_child_weight, subsample, colsample, reg = values
        params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample,
            **reg,
        }
        encoded = json.dumps(params, sort_keys=True, separators=(",", ":"))
        params["candidate_id"] = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:12]
        combinations.append(params)
    rng = random.Random(seed + 1009 * CASES.index(case))
    rng.shuffle(combinations)
    for sequence, params in enumerate(combinations, start=1):
        params["grid_sequence"] = sequence
    return combinations


def save_candidate_grid(path: Path, candidates: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(candidates)
    ordered = [
        "grid_sequence",
        "candidate_id",
        "learning_rate",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
        "gamma",
    ]
    atomic_csv(path, frame[ordered])


def load_case(data_root: Path, case: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info("%s: loading CAMS and observation datasets", case)
    cams_path = data_root / f"{case}data" / "CAMS_dataset.h5"
    obs_path = data_root / f"{case}data" / "obs_dataset.h5"
    cams = pd.read_hdf(cams_path)
    obs = pd.read_hdf(obs_path)

    required_cams = set(FEATURES) | {
        "ObservationPeriod",
        "station",
        case,
        "DHI",
        "TOA",
        "Snow_probability",
        "CloudOpticalDepth",
        "CloudType",
        "CloudCoverage",
    }
    required_obs = {"ObservationPeriod", "station", case, "DHI"}
    missing_cams = sorted(required_cams - set(cams.columns))
    missing_obs = sorted(required_obs - set(obs.columns))
    if missing_cams or missing_obs:
        raise ValueError(
            f"{case}: dataset schema differs from the example; "
            f"missing CAMS={missing_cams}, missing observations={missing_obs}"
        )

    cams = cams[cams["station"].isin(ACCEPTED_STATIONS)].copy()
    obs = obs[obs["station"].isin(ACCEPTED_STATIONS)].copy()
    key_columns = ["ObservationPeriod", "station"]
    if cams.duplicated(key_columns).any() or obs.duplicated(key_columns).any():
        raise ValueError(f"{case}: duplicate ObservationPeriod/station keys found")

    cams = cams.set_index(key_columns)
    obs = obs.set_index(key_columns)
    frame = pd.concat([cams, obs.add_suffix("_obs")], axis=1).reset_index()

    # Quality filters: SZA < 80 degrees, positive CAMS and observed irradiance,
    # valid cloud and snow fields. For GHI and BNI, positive DHI is also
    # required in both CAMS and observations.
    mask = (
        (frame["sza"] < 80.0)
        & (frame[case] > 0.0)
        & (frame[f"{case}_obs"] > 0.0)
        & (frame["TOA"] > 0.0)
        & (frame["Snow_probability"] >= 0.0)
        & frame["CloudOpticalDepth"].notna()
        & (frame["CloudType"] >= 0.0)
        & (frame["CloudCoverage"] >= 0.0)
    )
    if case in ("GHI", "BNI"):
        mask &= (frame["DHI"] > 0.0) & (frame["DHI_obs"] > 0.0)
    frame = frame.loc[mask].dropna(subset=[case, f"{case}_obs"]).copy()
    frame["ObservationPeriod"] = pd.to_datetime(frame["ObservationPeriod"])
    frame["date"] = frame["ObservationPeriod"].dt.strftime("%Y-%m-%d")
    frame.reset_index(drop=True, inplace=True)
    if frame.empty:
        raise ValueError(f"{case}: no rows remain after preprocessing")
    logger.info(
        "%s: %s usable rows, %s stations, %s station-date groups",
        case,
        f"{len(frame):,}",
        frame["station"].nunique(),
        frame[["station", "date"]].drop_duplicates().shape[0],
    )
    return frame


def make_or_load_splits(
    frame: pd.DataFrame,
    path: Path,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
    logger: logging.Logger,
) -> pd.Series:
    current_groups = (
        frame[["station", "date"]]
        .drop_duplicates()
        .sort_values(["station", "date"])
        .reset_index(drop=True)
    )
    if path.exists():
        groups = pd.read_csv(path, dtype={"station": str, "date": str, "split": str})
        expected_columns = {"station", "date", "split"}
        if set(groups.columns) != expected_columns:
            raise ValueError(f"Existing split file has unexpected columns: {path}")
        if groups.duplicated(["station", "date"]).any():
            raise ValueError(f"Existing split file has duplicate groups: {path}")
        left = set(map(tuple, current_groups[["station", "date"]].to_numpy()))
        right = set(map(tuple, groups[["station", "date"]].to_numpy()))
        if left != right:
            raise ValueError(
                f"Dataset groups changed since {path} was created. "
                "Use a new --output-dir to create a fresh, comparable split."
            )
        if not set(groups["split"]).issubset(SPLIT_NAMES):
            raise ValueError(f"Existing split file contains an invalid split name: {path}")
        logger.info("Reusing saved station-date split: %s", path)
    else:
        rng = np.random.default_rng(seed)
        split_parts = []
        # Split dates separately within each station so every sufficiently large
        # station is represented in all three partitions.
        for station, station_groups in current_groups.groupby("station", sort=True):
            station_groups = station_groups.copy().reset_index(drop=True)
            order = rng.permutation(len(station_groups))
            n_train = int(math.floor(train_fraction * len(station_groups)))
            n_validation = int(math.floor(validation_fraction * len(station_groups)))
            labels = np.full(len(station_groups), "test", dtype=object)
            labels[order[:n_train]] = "train"
            labels[order[n_train : n_train + n_validation]] = "validation"
            station_groups["split"] = labels
            split_parts.append(station_groups)
        groups = pd.concat(split_parts, ignore_index=True)
        atomic_csv(path, groups[["station", "date", "split"]])
        logger.info("Saved deterministic station-date split: %s", path)

    lookup = groups.set_index(["station", "date"])["split"]
    row_index = pd.MultiIndex.from_frame(frame[["station", "date"]])
    row_splits = pd.Series(row_index.map(lookup), index=frame.index, name="split")
    if row_splits.isna().any():
        raise RuntimeError("Some rows were not assigned to a data split")
    for split_name in SPLIT_NAMES:
        if not (row_splits == split_name).any():
            raise RuntimeError(f"The {split_name} split is empty")
    return row_splits


def radiation_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - true
    rmse = float(np.sqrt(np.mean(np.square(error))))
    mae = float(np.mean(np.abs(error)))
    median_bias = float(np.median(error))
    mean_bias = float(np.mean(error))
    denominator = float(np.sum(true))
    nmb = float(100.0 * np.sum(error) / denominator) if denominator else float("nan")
    nme = float(100.0 * np.sum(np.abs(error)) / denominator) if denominator else float("nan")
    centered = true - np.mean(true)
    total_sum_squares = float(np.sum(np.square(centered)))
    r2 = (
        float(1.0 - np.sum(np.square(error)) / total_sum_squares)
        if total_sum_squares
        else float("nan")
    )
    if len(true) > 1 and np.std(true) > 0 and np.std(predicted) > 0:
        pearson_r2 = float(np.corrcoef(true, predicted)[0, 1] ** 2)
    else:
        pearson_r2 = float("nan")
    return {
        "median_bias": median_bias,
        "mean_bias": mean_bias,
        "rmse": rmse,
        "mae": mae,
        "nmb_percent": nmb,
        "nme_percent": nme,
        "r2": r2,
        "pearson_r2": pearson_r2,
    }


def prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def make_dmatrices(
    frame: pd.DataFrame, splits: pd.Series, case: str, nthread: int
) -> tuple[dict[str, xgb.DMatrix], dict[str, dict[str, np.ndarray]]]:
    features = frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    correction = (frame[f"{case}_obs"] - frame[case]).to_numpy(dtype=np.float32)
    true = frame[f"{case}_obs"].to_numpy(dtype=np.float32)
    uncorrected = frame[case].to_numpy(dtype=np.float32)
    matrices: dict[str, xgb.DMatrix] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split_name in SPLIT_NAMES:
        mask = splits.to_numpy() == split_name
        split_features = features.loc[mask]
        split_correction = correction[mask]
        matrices[split_name] = xgb.DMatrix(
            split_features,
            label=split_correction,
            feature_names=FEATURES,
            missing=np.nan,
            nthread=nthread,
        )
        arrays[split_name] = {
            "correction": split_correction,
            "true": true[mask],
            "uncorrected": uncorrected[mask],
        }
    return matrices, arrays


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pd.read_csv(path, dtype={"candidate_id": str, "status": str})
    return frame.to_dict(orient="records")


def save_results(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for column in ("grid_sequence", "completed_at"):
            if column not in frame:
                frame[column] = np.nan
        frame.sort_values(["grid_sequence", "completed_at"], inplace=True, na_position="last")
    atomic_csv(path, frame)


def best_successful_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = []
    for row in rows:
        try:
            if row.get("status") == "ok" and np.isfinite(float(row["validation_corrected_rmse"])):
                successful.append(row)
        except (KeyError, TypeError, ValueError):
            continue
    return min(successful, key=lambda row: float(row["validation_corrected_rmse"]), default=None)


def train_candidate(
    candidate: dict[str, Any],
    matrices: dict[str, xgb.DMatrix],
    arrays: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> tuple[xgb.Booster, dict[str, Any], dict[str, list[float]]]:
    candidate_id = candidate["candidate_id"]
    grid_sequence = int(candidate["grid_sequence"])
    search_params = {
        key: value
        for key, value in candidate.items()
        if key not in {"candidate_id", "grid_sequence"}
    }
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_bin": 256,
        "nthread": args.nthread,
        "seed": args.seed,
        "verbosity": 0,
        "validate_parameters": True,
        **search_params,
    }
    evaluation_history: dict[str, dict[str, list[float]]] = {}
    callback = xgb.callback.EarlyStopping(
        rounds=args.early_stopping_rounds,
        metric_name="rmse",
        data_name="validation",
        maximize=False,
        save_best=True,
    )
    start = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=matrices["train"],
        num_boost_round=args.max_boost_rounds,
        evals=[(matrices["train"], "train"), (matrices["validation"], "validation")],
        evals_result=evaluation_history,
        callbacks=[callback],
        verbose_eval=False,
    )
    elapsed = time.perf_counter() - start

    row: dict[str, Any] = {
        "status": "ok",
        "candidate_id": candidate_id,
        "grid_sequence": grid_sequence,
        "completed_at": utc_now(),
        "fit_seconds": elapsed,
        "best_iteration_zero_based": int(booster.best_iteration),
        "best_num_boost_rounds": int(booster.best_iteration) + 1,
        **search_params,
    }
    for split_name in ("train", "validation"):
        predicted_correction = booster.predict(matrices[split_name])
        corrected = arrays[split_name]["uncorrected"] + predicted_correction
        corrected_metrics = radiation_metrics(arrays[split_name]["true"], corrected)
        baseline_metrics = radiation_metrics(
            arrays[split_name]["true"], arrays[split_name]["uncorrected"]
        )
        row.update(prefixed(corrected_metrics, f"{split_name}_corrected"))
        row.update(prefixed(baseline_metrics, f"{split_name}_baseline"))
    baseline_rmse = row["validation_baseline_rmse"]
    row["validation_rmse_improvement_percent"] = (
        100.0 * (baseline_rmse - row["validation_corrected_rmse"]) / baseline_rmse
        if baseline_rmse
        else float("nan")
    )
    # A concise history is sufficient for the selected model's learning curve.
    history = {
        split_name: [float(value) for value in metrics["rmse"]]
        for split_name, metrics in evaluation_history.items()
    }
    return booster, row, history


def save_best_artifacts(
    case_dir: Path,
    booster: xgb.Booster,
    row: dict[str, Any],
    history: dict[str, list[float]],
) -> None:
    model_path = case_dir / "best_model.json"
    temporary_model = case_dir / "best_model.tmp.json"
    booster.save_model(temporary_model)
    os.replace(temporary_model, model_path)
    atomic_json(case_dir / "best_candidate.json", row)
    atomic_json(case_dir / "best_learning_history.json", history)


def load_best_booster(case_dir: Path, nthread: int) -> xgb.Booster:
    booster = xgb.Booster({"nthread": nthread})
    booster.load_model(case_dir / "best_model.json")
    return booster


def color_map() -> LinearSegmentedColormap:
    colors_8bit = [
        (255, 255, 255),
        (127, 188, 227),
        (82, 170, 115),
        (225, 189, 74),
        (222, 64, 39),
        (149, 21, 25),
    ]
    colors = [tuple(channel / 255.0 for channel in color) for color in colors_8bit]
    return LinearSegmentedColormap.from_list("radiation_density", colors, N=256)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_performance(
    case: str,
    true: np.ndarray,
    uncorrected: np.ndarray,
    corrected: np.ndarray,
    figures_dir: Path,
) -> None:
    maximum = PLOT_MAX[case]
    baseline = radiation_metrics(true, uncorrected)
    corrected_metrics = radiation_metrics(true, corrected)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), sharex=True, sharey=True)
    for axis, predicted, metrics, title, ylabel in [
        (axes[0], uncorrected, baseline, "Uncorrected", f"CAMS {case} (W/m²)"),
        (
            axes[1],
            corrected,
            corrected_metrics,
            "XGBoost corrected",
            f"Bias-corrected CAMS {case} (W/m²)",
        ),
    ]:
        axis.hist2d(
            true,
            predicted,
            bins=[80, 120],
            range=[[0, maximum], [0, maximum]],
            cmap=color_map(),
            norm=LogNorm(),
            alpha=0.85,
        )
        axis.plot([0, maximum], [0, maximum], color="black", linewidth=1.0)
        axis.grid(True, alpha=0.25)
        axis.set_xlim(0, maximum)
        axis.set_ylim(0, maximum)
        axis.set_title(title)
        axis.set_xlabel(f"Observed {case} (W/m²)")
        axis.set_ylabel(ylabel)
        axis.text(
            0.03,
            0.97,
            f"Median bias: {metrics['median_bias']:.1f} W/m²\n"
            f"RMSE: {metrics['rmse']:.1f} W/m²\n"
            f"R²: {metrics['r2']:.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.95},
        )
    fig.suptitle(f"{case} held-out test performance")
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{case}_test_performance")


def plot_learning_curve(case: str, history: dict[str, list[float]], figures_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for split_name, values in history.items():
        axis.plot(np.arange(1, len(values) + 1), values, label=split_name.capitalize())
    axis.set_xlabel("Boosting round")
    axis.set_ylabel("Correction RMSE (W/m²)")
    axis.set_title(f"{case} best-model learning curve")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{case}_learning_curve")


def plot_feature_importance(case: str, booster: xgb.Booster, figures_dir: Path) -> None:
    scores = booster.get_score(importance_type="gain")
    values = pd.Series({feature: float(scores.get(feature, 0.0)) for feature in FEATURES})
    values = values.sort_values(ascending=True)
    total = values.sum()
    if total > 0:
        values = 100.0 * values / total
    fig, axis = plt.subplots(figsize=(7.2, 5.8))
    axis.barh(values.index, values.values, color="#4C78A8")
    axis.set_xlabel("Share of total gain (%)")
    axis.set_title(f"{case} best-model feature importance")
    axis.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{case}_feature_importance")


def plot_search_overview(case: str, rows: list[dict[str, Any]], figures_dir: Path) -> None:
    frame = pd.DataFrame(rows)
    if frame.empty or "status" not in frame:
        return
    frame = frame[frame["status"] == "ok"].copy()
    if frame.empty:
        return
    frame["validation_corrected_rmse"] = pd.to_numeric(
        frame["validation_corrected_rmse"], errors="coerce"
    )
    frame["grid_sequence"] = pd.to_numeric(frame["grid_sequence"], errors="coerce")
    frame["max_depth"] = pd.to_numeric(frame["max_depth"], errors="coerce")
    frame.dropna(subset=["validation_corrected_rmse", "grid_sequence"], inplace=True)
    frame.sort_values("grid_sequence", inplace=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    scatter = axes[0].scatter(
        frame["grid_sequence"],
        frame["validation_corrected_rmse"],
        c=frame["max_depth"],
        cmap="viridis",
        s=18,
        alpha=0.75,
    )
    axes[0].axhline(
        frame["validation_corrected_rmse"].min(), color="black", linestyle="--", linewidth=1
    )
    axes[0].set_xlabel("Grid sequence")
    axes[0].set_ylabel("Validation RMSE (W/m²)")
    axes[0].set_title("All completed candidates")
    axes[0].grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=axes[0], label="max_depth")

    top = frame.nsmallest(min(20, len(frame)), "validation_corrected_rmse").copy()
    top.sort_values("validation_corrected_rmse", ascending=True, inplace=True)
    labels = [str(value)[:8] for value in top["candidate_id"]]
    axes[1].barh(labels, top["validation_corrected_rmse"], color="#59A14F")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Validation RMSE (W/m²)")
    axes[1].set_ylabel("Candidate ID")
    axes[1].set_title("Best completed candidates")
    axes[1].grid(True, axis="x", alpha=0.2)
    fig.suptitle(f"{case} grid-search overview")
    fig.tight_layout()
    save_figure(fig, figures_dir / f"{case}_grid_search_overview")


def write_case_summary(
    case: str,
    case_dir: Path,
    best_row: dict[str, Any],
    test_payload: dict[str, Any],
    completed: int,
    total: int,
) -> None:
    search_params = {
        key: best_row[key]
        for key in (
            "learning_rate",
            "max_depth",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "gamma",
        )
    }
    lines = [
        f"{case} XGBoost grid-search summary",
        "=" * (len(case) + 30),
        f"Search progress: {completed}/{total} candidates completed",
        f"Best candidate: {best_row['candidate_id']}",
        f"Best boosting rounds: {int(float(best_row['best_num_boost_rounds']))}",
        f"Validation corrected RMSE: {float(best_row['validation_corrected_rmse']):.3f} W/m²",
        f"Test baseline RMSE: {test_payload['baseline_metrics']['rmse']:.3f} W/m²",
        f"Test corrected RMSE: {test_payload['corrected_metrics']['rmse']:.3f} W/m²",
        f"Test RMSE improvement: {test_payload['rmse_improvement_percent']:.2f}%",
        "",
        "Best hyperparameters:",
    ]
    lines.extend(f"  {key}: {value}" for key, value in search_params.items())
    lines.append("")
    (case_dir / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


def finalize_case(
    case: str,
    case_dir: Path,
    rows: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    matrices: dict[str, xgb.DMatrix],
    arrays: dict[str, dict[str, np.ndarray]],
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    best_row = best_successful_row(rows)
    model_path = case_dir / "best_model.json"
    meta_path = case_dir / "best_candidate.json"
    if best_row is None or not model_path.exists() or not meta_path.exists():
        logger.warning("%s: no successful model is available for final evaluation", case)
        return None
    saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if str(saved_meta.get("candidate_id")) != str(best_row.get("candidate_id")):
        raise RuntimeError(
            f"{case}: best-model metadata does not match the best statistics row. "
            "Rerun with --retry-errors or use a new output directory."
        )

    booster = load_best_booster(case_dir, args.nthread)
    predicted_correction = booster.predict(matrices["test"])
    corrected = arrays["test"]["uncorrected"] + predicted_correction
    baseline_metrics = radiation_metrics(arrays["test"]["true"], arrays["test"]["uncorrected"])
    corrected_metrics = radiation_metrics(arrays["test"]["true"], corrected)
    improvement = (
        100.0 * (baseline_metrics["rmse"] - corrected_metrics["rmse"]) / baseline_metrics["rmse"]
        if baseline_metrics["rmse"]
        else float("nan")
    )
    completed_ids = {
        str(row.get("candidate_id"))
        for row in rows
        if row.get("status") in {"ok", "error"}
    }
    test_payload = {
        "case": case,
        "evaluated_at": utc_now(),
        "candidate_id": str(best_row["candidate_id"]),
        "test_rows": int(matrices["test"].num_row()),
        "baseline_metrics": baseline_metrics,
        "corrected_metrics": corrected_metrics,
        "rmse_improvement_percent": improvement,
        "search_candidates_completed": len(completed_ids),
        "search_candidates_total": len(candidates),
        "search_complete": len(completed_ids) == len(candidates),
    }
    atomic_json(case_dir / "test_metrics.json", test_payload)
    figures_dir = case_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_performance(
        case,
        arrays["test"]["true"],
        arrays["test"]["uncorrected"],
        corrected,
        figures_dir,
    )
    history = json.loads((case_dir / "best_learning_history.json").read_text(encoding="utf-8"))
    plot_learning_curve(case, history, figures_dir)
    plot_feature_importance(case, booster, figures_dir)
    plot_search_overview(case, rows, figures_dir)
    write_case_summary(
        case,
        case_dir,
        best_row,
        test_payload,
        len(completed_ids),
        len(candidates),
    )
    logger.info(
        "%s: test RMSE %.3f -> %.3f W/m² (%.2f%% improvement)",
        case,
        baseline_metrics["rmse"],
        corrected_metrics["rmse"],
        improvement,
    )
    return {
        "case": case,
        "best_candidate_id": str(best_row["candidate_id"]),
        "validation_rmse": float(best_row["validation_corrected_rmse"]),
        "test_baseline_rmse": baseline_metrics["rmse"],
        "test_corrected_rmse": corrected_metrics["rmse"],
        "test_rmse_improvement_percent": improvement,
        "candidates_completed": len(completed_ids),
        "candidates_total": len(candidates),
        "search_complete": len(completed_ids) == len(candidates),
    }


def process_case(
    case: str,
    case_deadline: float,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    case_dir = args.output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    candidates = candidate_grid(args.seed, case)
    if args.max_candidates_per_case is not None:
        candidates = candidates[: args.max_candidates_per_case]
    save_candidate_grid(case_dir / "candidate_grid.csv", candidates)

    frame = load_case(args.data_root, case, logger)
    splits = make_or_load_splits(
        frame,
        case_dir / "station_date_splits.csv",
        args.seed,
        args.train_fraction,
        args.validation_fraction,
        logger,
    )
    split_summary = {
        split_name: {
            "rows": int((splits == split_name).sum()),
            "groups": int(
                frame.loc[splits == split_name, ["station", "date"]].drop_duplicates().shape[0]
            ),
            "stations": int(frame.loc[splits == split_name, "station"].nunique()),
        }
        for split_name in SPLIT_NAMES
    }
    atomic_json(
        case_dir / "data_summary.json",
        {
            "case": case,
            "usable_rows": len(frame),
            "features": FEATURES,
            "split_summary": split_summary,
            "target": f"{case}_obs - {case}",
        },
    )
    matrices, arrays = make_dmatrices(frame, splits, case, args.nthread)
    del frame, splits

    results_path = case_dir / "search_results.csv"
    rows = load_results(results_path)
    completed_statuses = {"ok"} if args.retry_errors else {"ok", "error"}
    completed_ids = {
        str(row.get("candidate_id")) for row in rows if row.get("status") in completed_statuses
    }
    current_best = best_successful_row(rows)
    current_best_rmse = (
        float(current_best["validation_corrected_rmse"])
        if current_best is not None
        else float("inf")
    )
    pending = [c for c in candidates if c["candidate_id"] not in completed_ids]
    logger.info(
        "%s: %d/%d candidates already recorded; %d pending",
        case,
        len(candidates) - len(pending),
        len(candidates),
        len(pending),
    )
    if args.dry_run:
        logger.info("%s: dry run; skipping model fits", case)
        return None

    fitted_now = 0
    for candidate in pending:
        if fitted_now > 0 and time.monotonic() >= case_deadline:
            logger.info("%s: case time allocation reached; saving partial progress", case)
            break
        fitted_now += 1
        candidate_id = candidate["candidate_id"]
        logger.info(
            "%s: fitting candidate %s (grid %d/%d, new fit %d)",
            case,
            candidate_id,
            candidate["grid_sequence"],
            len(candidates),
            fitted_now,
        )
        try:
            booster, row, history = train_candidate(candidate, matrices, arrays, args)
            validation_rmse = float(row["validation_corrected_rmse"])
            if validation_rmse < current_best_rmse:
                save_best_artifacts(case_dir, booster, row, history)
                current_best_rmse = validation_rmse
                logger.info(
                    "%s: new best %s, validation RMSE %.3f W/m², %d rounds",
                    case,
                    candidate_id,
                    validation_rmse,
                    row["best_num_boost_rounds"],
                )
            else:
                logger.info(
                    "%s: candidate %s validation RMSE %.3f W/m² (%.1f s)",
                    case,
                    candidate_id,
                    validation_rmse,
                    row["fit_seconds"],
                )
            # Replace a previous error row if --retry-errors was used.
            rows = [r for r in rows if str(r.get("candidate_id")) != candidate_id]
            rows.append(row)
            save_results(results_path, rows)
        except KeyboardInterrupt:
            logger.warning("Interrupted during %s candidate %s", case, candidate_id)
            raise
        except Exception as exc:
            logger.error("%s candidate %s failed: %s", case, candidate_id, exc)
            logger.debug(traceback.format_exc())
            error_row = {
                "status": "error",
                "candidate_id": candidate_id,
                "grid_sequence": candidate["grid_sequence"],
                "completed_at": utc_now(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                **{
                    key: value
                    for key, value in candidate.items()
                    if key not in {"candidate_id", "grid_sequence"}
                },
            }
            rows = [r for r in rows if str(r.get("candidate_id")) != candidate_id]
            rows.append(error_row)
            save_results(results_path, rows)

    return finalize_case(case, case_dir, rows, candidates, matrices, arrays, args, logger)


def write_overall_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    frame = pd.DataFrame(summaries)
    atomic_csv(output_dir / "overall_summary.csv", frame)
    lines = ["XGBoost radiation error-correction grid search", "=" * 46, ""]
    for summary in summaries:
        completion = (
            "complete" if summary["search_complete"] else "partial; rerun the same command to resume"
        )
        lines.extend(
            [
                f"{summary['case']}:",
                f"  Search: {summary['candidates_completed']}/{summary['candidates_total']} ({completion})",
                f"  Best candidate: {summary['best_candidate_id']}",
                f"  Validation RMSE: {summary['validation_rmse']:.3f} W/m²",
                f"  Test RMSE: {summary['test_baseline_rmse']:.3f} -> {summary['test_corrected_rmse']:.3f} W/m²",
                f"  Test improvement: {summary['test_rmse_improvement_percent']:.2f}%",
                "",
            ]
        )
    (output_dir / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output_dir)
    validate_input_files(args.data_root, args.cases)
    grid_size = len(candidate_grid(args.seed, args.cases[0]))
    if args.max_candidates_per_case is not None:
        grid_size = min(grid_size, args.max_candidates_per_case)
    logger.info("Data root: %s", args.data_root)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Python: %s", sys.executable)
    logger.info("XGBoost %s; pandas %s; NumPy %s", xgb.__version__, pd.__version__, np.__version__)
    logger.info("Using %d CPU threads; %d candidates per case", args.nthread, grid_size)
    logger.info("Maximum total runtime: %.2f hours", args.max_hours)
    atomic_json(
        args.output_dir / "run_config.json",
        {
            "updated_at": utc_now(),
            "command": sys.argv,
            "arguments": vars(args),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "xgboost_version": xgb.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "candidates_per_case": grid_size,
            "resume_behavior": "Successful fits are skipped; errors are skipped unless --retry-errors is used.",
        },
    )
    if args.dry_run:
        logger.info("Dry-run validation started")

    sleep_process = prevent_macos_sleep(logger, args.allow_sleep)
    overall_start = time.monotonic()
    global_deadline = overall_start + args.max_hours * 3600.0
    summaries: list[dict[str, Any]] = []
    try:
        for case_number, case in enumerate(args.cases, start=1):
            cases_remaining = len(args.cases) - case_number + 1
            seconds_remaining = max(0.0, global_deadline - time.monotonic())
            # Divide the remaining time evenly so a slow first case cannot starve
            # DHI or BNI. Unused time automatically rolls forward.
            case_deadline = time.monotonic() + seconds_remaining / cases_remaining
            logger.info(
                "%s: allocated up to %.2f hours in this run",
                case,
                max(0.0, case_deadline - time.monotonic()) / 3600.0,
            )
            summary = process_case(case, case_deadline, args, logger)
            if summary is not None:
                summaries.append(summary)
                write_overall_summary(args.output_dir, summaries)
        elapsed_hours = (time.monotonic() - overall_start) / 3600.0
        logger.info("Run finished in %.2f hours", elapsed_hours)
        if args.dry_run:
            logger.info("Dry run successful; no models were fitted")
        return 0
    except KeyboardInterrupt:
        logger.warning("Run stopped by user. Completed candidates are saved; rerun to resume.")
        write_overall_summary(args.output_dir, summaries)
        return 130
    finally:
        if sleep_process is not None and sleep_process.poll() is None:
            sleep_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
