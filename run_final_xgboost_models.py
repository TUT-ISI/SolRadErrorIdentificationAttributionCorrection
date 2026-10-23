#!/usr/bin/env python3
"""Fit and evaluate final GHI, DHI, and BNI XGBoost correction models.

This script consumes the completed outputs from run_xgboost_grid_search.py.
For each radiation case it:

1. Loads the frozen best hyperparameters and best boosting-round count.
2. Refits one final model on the combined training + validation partitions.
3. Evaluates the refitted model on the unchanged held-out test partition.
4. Saves the model, test predictions, metrics, and publication-ready figures.

With --leave-one-station-out, the same frozen hyperparameters and boosting-
round count are used for station-level cross-validation. Each fold trains on
all usable observations from every other station and evaluates on every usable
observation from the omitted station. This is a non-nested LOSO analysis:
hyperparameters were selected once by the preceding station-date grid search.

No evaluation result is used to change a hyperparameter or boosting-round count.


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
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cameo_final_xgb_mpl_cache")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import xgboost as xgb


CASES = ("GHI", "DHI", "BNI")
SEARCH_PARAMETER_NAMES = (
    "learning_rate",
    "max_depth",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "gamma",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit selected GHI, DHI, and BNI models for either the final held-out "
            "station-date test or leave-one-station-out evaluation."
        )
    )
    default_root = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_root,
        help="Folder containing the case datasets and grid-search script.",
    )
    parser.add_argument(
        "--search-results",
        type=Path,
        default=None,
        help="Completed grid-search results folder (default: DATA_ROOT/grid_search_results).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Results folder (default: DATA_ROOT/final_model_results, or "
            "DATA_ROOT/leave_one_station_out_results in LOSO mode)."
        ),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=CASES,
        default=list(CASES),
        help="Cases to run (default: GHI DHI BNI).",
    )
    parser.add_argument(
        "--nthread",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="CPU threads for XGBoost (default: all logical CPUs except one).",
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--leave-one-station-out",
        action="store_true",
        help=(
            "Run one fold per station: train on all other stations and evaluate "
            "on all rows from the omitted station."
        ),
    )
    parser.add_argument(
        "--stations",
        nargs="+",
        default=None,
        help=(
            "Optional station IDs to run in LOSO mode, useful for testing or "
            "resuming selected folds. Default: every available station."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute completed station folds in LOSO mode.",
    )
    parser.add_argument(
        "--allow-sleep",
        action="store_true",
        help="Do not keep macOS awake during final fitting.",
    )
    args = parser.parse_args()
    args.data_root = args.data_root.expanduser().resolve()
    args.search_results = (
        args.search_results.expanduser().resolve()
        if args.search_results is not None
        else args.data_root / "grid_search_results"
    )
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.data_root
        / (
            "leave_one_station_out_results"
            if args.leave_one_station_out
            else "final_model_results"
        )
    )
    if args.nthread <= 0:
        parser.error("--nthread must be positive")
    if args.stations is not None and not args.leave_one_station_out:
        parser.error("--stations is only valid with --leave-one-station-out")
    if args.force and not args.leave_one_station_out:
        parser.error("--force is currently only valid with --leave-one-station-out")
    return args


def setup_logging(output_dir: Path, leave_one_station_out: bool) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cameo_final_xgboost")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    log_name = "loso_training.log" if leave_one_station_out else "final_training.log"
    log_mode = "a" if leave_one_station_out else "w"
    file_handler = logging.FileHandler(output_dir / log_name, mode=log_mode)
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
        logger.info("Idle system sleep is disabled until final fitting is complete.")
        return process
    except OSError as exc:
        logger.warning("Could not start caffeinate; the Mac may sleep: %s", exc)
        return None


def load_grid_module(data_root: Path):
    script_path = data_root / "run_xgboost_grid_search.py"
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required companion script not found: {script_path}. "
            "Keep this final script in the same folder as run_xgboost_grid_search.py."
        )
    specification = importlib.util.spec_from_file_location("cameo_grid_search", script_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def validate_search_case(search_results: Path, case: str) -> tuple[dict[str, Any], dict[str, Any]]:
    case_dir = search_results / case
    required = [
        case_dir / "best_candidate.json",
        case_dir / "test_metrics.json",
        case_dir / "station_date_splits.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing completed search output(s):\n" + "\n".join(missing))
    best = json.loads((case_dir / "best_candidate.json").read_text(encoding="utf-8"))
    search_test = json.loads((case_dir / "test_metrics.json").read_text(encoding="utf-8"))
    if not search_test.get("search_complete", False):
        raise RuntimeError(
            f"{case}: grid search is marked incomplete. Resume the grid search before final fitting."
        )
    if str(best.get("candidate_id")) != str(search_test.get("candidate_id")):
        raise RuntimeError(f"{case}: best-candidate and test-metrics identifiers do not match")
    missing_params = [name for name in SEARCH_PARAMETER_NAMES if name not in best]
    if missing_params or "best_num_boost_rounds" not in best:
        raise ValueError(f"{case}: selected-model metadata is incomplete: {missing_params}")
    return best, search_test


def percent_improvement(baseline: float, corrected: float) -> float:
    return 100.0 * (baseline - corrected) / baseline if baseline else float("nan")


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
        "mae": mae,
        "rmse": rmse,
        "nmb_percent": nmb,
        "nme_percent": nme,
        "r2": r2,
        "pearson_r2": pearson_r2,
    }


def prepare_model_data(
    frame: pd.DataFrame,
    case: str,
    grid_module,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Create exactly the feature matrix used by this final-model script."""
    base_features = list(grid_module.FEATURES)
    model_features = base_features + [case]
    all_features = (
        frame[base_features].replace([np.inf, -np.inf], np.nan).astype(np.float32)
    )
    # Retain the current final-model convention: the uncorrected radiation
    # component is an additional predictor of its own correction.
    all_features[case] = frame[case].to_numpy(dtype=np.float32)
    correction_target = (frame[f"{case}_obs"] - frame[case]).to_numpy(dtype=np.float32)
    return all_features, correction_target, model_features


def final_training_parameters(
    best: dict[str, Any],
    seed: int,
    nthread: int,
) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "max_bin": 256,
        "nthread": nthread,
        "seed": seed,
        "verbosity": 0,
        "validate_parameters": True,
        **{name: best[name] for name in SEARCH_PARAMETER_NAMES},
    }


def feature_gain_table(
    booster: xgb.Booster,
    model_features: list[str],
) -> pd.DataFrame:
    gain = booster.get_score(importance_type="gain")
    total_gain = float(sum(gain.values()))
    rows = [
        {
            "feature": feature,
            "gain": float(gain.get(feature, 0.0)),
            "gain_percent": (
                100.0 * float(gain.get(feature, 0.0)) / total_gain if total_gain else 0.0
            ),
        }
        for feature in model_features
    ]
    result = pd.DataFrame(rows).sort_values("gain_percent", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result


def fit_final_case(
    case: str,
    args: argparse.Namespace,
    grid_module,
    logger: logging.Logger,
) -> dict[str, Any]:
    best, search_test = validate_search_case(args.search_results, case)
    logger.info(
        "%s: selected candidate %s with %d boosting rounds",
        case,
        best["candidate_id"],
        int(best["best_num_boost_rounds"]),
    )
    frame = grid_module.load_case(args.data_root, case, logger)
    split_path = args.search_results / case / "station_date_splits.csv"
    row_splits = grid_module.make_or_load_splits(
        frame,
        split_path,
        args.seed,
        0.70,
        0.15,
        logger,
    )
    train_mask = row_splits.isin(["train", "validation"]).to_numpy()
    test_mask = (row_splits == "test").to_numpy()
    all_features, correction_target, model_features = prepare_model_data(
        frame, case, grid_module
    )

    dtrain = xgb.DMatrix(
        all_features.loc[train_mask],
        label=correction_target[train_mask],
        feature_names=model_features,
        missing=np.nan,
        nthread=args.nthread,
    )
    dtest = xgb.DMatrix(
        all_features.loc[test_mask],
        label=correction_target[test_mask],
        feature_names=model_features,
        missing=np.nan,
        nthread=args.nthread,
    )
    params = final_training_parameters(best, args.seed, args.nthread)
    num_boost_rounds = int(best["best_num_boost_rounds"])
    fit_start = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_rounds,
        verbose_eval=False,
    )
    fit_seconds = time.perf_counter() - fit_start

    train_correction = booster.predict(dtrain)
    test_correction = booster.predict(dtest)
    train_frame = frame.loc[train_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    train_true = train_frame[f"{case}_obs"].to_numpy(dtype=np.float64)
    train_uncorrected = train_frame[case].to_numpy(dtype=np.float64)
    train_corrected = train_uncorrected + train_correction
    test_true = test_frame[f"{case}_obs"].to_numpy(dtype=np.float64)
    test_uncorrected = test_frame[case].to_numpy(dtype=np.float64)
    test_corrected = test_uncorrected + test_correction

    train_baseline_metrics = radiation_metrics(train_true, train_uncorrected)
    train_corrected_metrics = radiation_metrics(train_true, train_corrected)
    test_baseline_metrics = radiation_metrics(test_true, test_uncorrected)
    test_corrected_metrics = radiation_metrics(test_true, test_corrected)
    rmse_improvement = percent_improvement(
        test_baseline_metrics["rmse"], test_corrected_metrics["rmse"]
    )
    mae_improvement = percent_improvement(
        test_baseline_metrics["mae"], test_corrected_metrics["mae"]
    )

    case_dir = args.output_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    model_path = case_dir / "final_model.json"
    temporary_model = case_dir / "final_model.tmp.json"
    booster.save_model(temporary_model)
    os.replace(temporary_model, model_path)

    predictions = test_frame[["ObservationPeriod", "station", "date"]].copy()
    predictions[f"observed_{case}"] = test_true
    predictions[f"CAMS_{case}"] = test_uncorrected
    predictions[f"corrected_{case}"] = test_corrected
    predictions["uncorrected_error"] = test_uncorrected - test_true
    predictions["corrected_error"] = test_corrected - test_true
    atomic_csv(case_dir / "test_predictions.csv", predictions)

    # Per-station RMSE and relative improvement
    station_groups = predictions.groupby("station")
    station_rmse_improvement = {}
    for station, group in station_groups:
        baseline_rmse = np.sqrt(np.mean((group[f"CAMS_{case}"] - group[f"observed_{case}"]) ** 2))
        corrected_rmse = np.sqrt(np.mean((group[f"corrected_{case}"] - group[f"observed_{case}"]) ** 2))
        station_rmse_improvement[station] = {
            "baseline_rmse": baseline_rmse,
            "corrected_rmse": corrected_rmse,
            "rmse_improvement": percent_improvement(baseline_rmse, corrected_rmse),
        }
    atomic_json(case_dir / "station_rmse_improvement.json", station_rmse_improvement)

    gain_table = feature_gain_table(booster, model_features)
    normalized_gain = dict(zip(gain_table["feature"], gain_table["gain_percent"]))
    atomic_csv(case_dir / "feature_gain.csv", gain_table)
    metrics_payload = {
        "case": case,
        "created_at": utc_now(),
        "candidate_id": str(best["candidate_id"]),
        "num_boost_rounds": num_boost_rounds,
        "fit_seconds": fit_seconds,
        "training_rows": int(dtrain.num_row()),
        "test_rows": int(dtest.num_row()),
        "training_partitions": ["train", "validation"],
        "test_partition": "test",
        "model_features": model_features,
        "hyperparameters": {name: best[name] for name in SEARCH_PARAMETER_NAMES},
        "training_baseline_metrics": train_baseline_metrics,
        "training_corrected_metrics": train_corrected_metrics,
        "test_baseline_metrics": test_baseline_metrics,
        "test_corrected_metrics": test_corrected_metrics,
        "test_rmse_improvement_percent": rmse_improvement,
        "test_mae_improvement_percent": mae_improvement,
        "grid_selected_model_test_rmse": float(search_test["corrected_metrics"]["rmse"]),
        "final_refit_test_rmse_change": (
            test_corrected_metrics["rmse"] - float(search_test["corrected_metrics"]["rmse"])
        ),
        "feature_gain_percent": normalized_gain,
    }
    atomic_json(case_dir / "final_metrics.json", metrics_payload)
    make_publication_figure(
        case,
        test_true,
        test_uncorrected,
        test_corrected,
        args.output_dir / "figures",
        grid_module,
    )
    logger.info(
        "%s: final test RMSE %.3f -> %.3f W/m² (%.2f%%); MAE %.3f -> %.3f W/m²",
        case,
        test_baseline_metrics["rmse"],
        test_corrected_metrics["rmse"],
        rmse_improvement,
        test_baseline_metrics["mae"],
        test_corrected_metrics["mae"],
    )

    # Station-mean baseline: correct the test rows with the per-station mean
    # residual of the training partition and plot it like the XGBoost correction.
    mean_residuals_for_station = train_frame.groupby('station').apply(lambda x: (x[case + '_obs'] - x[case]).mean())
    test_corrected_based_on_mean_residuals = test_frame['station'].map(mean_residuals_for_station).values + test_frame[case].values
    make_publication_figure(
        case,
        test_true,
        test_uncorrected,
        test_corrected_based_on_mean_residuals,
        args.output_dir / "figures_mean_residuals",
        grid_module,
    )

    return {
        "case": case,
        "candidate_id": str(best["candidate_id"]),
        "num_boost_rounds": num_boost_rounds,
        "training_rows": int(dtrain.num_row()),
        "test_rows": int(dtest.num_row()),
        "baseline_mean_bias_W_m2": test_baseline_metrics["mean_bias"],
        "corrected_mean_bias_W_m2": test_corrected_metrics["mean_bias"],
        "baseline_median_bias_W_m2": test_baseline_metrics["median_bias"],
        "corrected_median_bias_W_m2": test_corrected_metrics["median_bias"],
        "baseline_MAE_W_m2": test_baseline_metrics["mae"],
        "corrected_MAE_W_m2": test_corrected_metrics["mae"],
        "MAE_improvement_percent": mae_improvement,
        "baseline_RMSE_W_m2": test_baseline_metrics["rmse"],
        "corrected_RMSE_W_m2": test_corrected_metrics["rmse"],
        "RMSE_improvement_percent": rmse_improvement,
        "baseline_R2": test_baseline_metrics["r2"],
        "corrected_R2": test_corrected_metrics["r2"],
        "grid_model_test_RMSE_W_m2": float(search_test["corrected_metrics"]["rmse"]),
        "final_refit_RMSE_change_W_m2": metrics_payload["final_refit_test_rmse_change"],
        "fit_seconds": fit_seconds,
    }


def loso_summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    baseline = metrics["test_baseline_metrics"]
    corrected = metrics["test_corrected_metrics"]
    return {
        "case": metrics["case"],
        "held_out_station": metrics["held_out_station"],
        "candidate_id": metrics["candidate_id"],
        "num_boost_rounds": metrics["num_boost_rounds"],
        "training_stations": metrics["training_stations"],
        "training_rows": metrics["training_rows"],
        "test_rows": metrics["test_rows"],
        "baseline_mean_bias_W_m2": baseline["mean_bias"],
        "corrected_mean_bias_W_m2": corrected["mean_bias"],
        "baseline_median_bias_W_m2": baseline["median_bias"],
        "corrected_median_bias_W_m2": corrected["median_bias"],
        "baseline_MAE_W_m2": baseline["mae"],
        "corrected_MAE_W_m2": corrected["mae"],
        "MAE_improvement_percent": metrics["test_mae_improvement_percent"],
        "baseline_RMSE_W_m2": baseline["rmse"],
        "corrected_RMSE_W_m2": corrected["rmse"],
        "RMSE_improvement_percent": metrics["test_rmse_improvement_percent"],
        "baseline_R2": baseline["r2"],
        "corrected_R2": corrected["r2"],
        "baseline_pearson_R2": baseline["pearson_r2"],
        "corrected_pearson_R2": corrected["pearson_r2"],
        "fit_seconds": metrics["fit_seconds"],
    }


def fit_loso_station(
    case: str,
    station: str,
    frame: pd.DataFrame,
    all_features: pd.DataFrame,
    correction_target: np.ndarray,
    model_features: list[str],
    best: dict[str, Any],
    args: argparse.Namespace,
    grid_module,
    logger: logging.Logger,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if Path(station).name != station:
        raise ValueError(f"Unsafe station identifier cannot be used as a directory: {station!r}")
    station_dir = args.output_dir / case / "stations" / station
    metrics_path = station_dir / "metrics.json"
    predictions_path = station_dir / "predictions.csv"
    required_outputs = [
        metrics_path,
        predictions_path,
        station_dir / "model.json",
        station_dir / "feature_gain.csv",
        station_dir / "figures" / f"{case}_corrected.pdf",
        station_dir / "figures" / f"{case}_corrected.png",
    ]
    if not args.force and all(path.is_file() for path in required_outputs):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("held_out_station") != station or metrics.get("case") != case:
            raise RuntimeError(f"Saved fold metadata does not match {case}/{station}")
        predictions = pd.read_csv(predictions_path, parse_dates=["ObservationPeriod"])
        logger.info("%s/%s: reusing completed station fold", case, station)
        return loso_summary_row(metrics), predictions

    train_mask = (frame["station"] != station).to_numpy()
    test_mask = (frame["station"] == station).to_numpy()
    if not test_mask.any() or not train_mask.any():
        raise RuntimeError(f"{case}/{station}: empty training or test partition")
    training_station_count = int(frame.loc[train_mask, "station"].nunique())
    dtrain = xgb.DMatrix(
        all_features.loc[train_mask],
        label=correction_target[train_mask],
        feature_names=model_features,
        missing=np.nan,
        nthread=args.nthread,
    )
    dtest = xgb.DMatrix(
        all_features.loc[test_mask],
        label=correction_target[test_mask],
        feature_names=model_features,
        missing=np.nan,
        nthread=args.nthread,
    )
    params = final_training_parameters(best, args.seed, args.nthread)
    num_boost_rounds = int(best["best_num_boost_rounds"])
    logger.info(
        "%s/%s: fitting on %s rows from %d other stations; testing on %s rows",
        case,
        station,
        f"{int(dtrain.num_row()):,}",
        training_station_count,
        f"{int(dtest.num_row()):,}",
    )
    fit_started = time.perf_counter()
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_rounds,
        verbose_eval=False,
    )
    fit_seconds = time.perf_counter() - fit_started
    train_correction = booster.predict(dtrain)
    test_correction = booster.predict(dtest)

    train_frame = frame.loc[train_mask]
    test_frame = frame.loc[test_mask]
    train_true = train_frame[f"{case}_obs"].to_numpy(dtype=np.float64)
    train_uncorrected = train_frame[case].to_numpy(dtype=np.float64)
    test_true = test_frame[f"{case}_obs"].to_numpy(dtype=np.float64)
    test_uncorrected = test_frame[case].to_numpy(dtype=np.float64)
    train_corrected = train_uncorrected + train_correction
    test_corrected = test_uncorrected + test_correction
    train_baseline = radiation_metrics(train_true, train_uncorrected)
    train_corrected_metrics = radiation_metrics(train_true, train_corrected)
    test_baseline = radiation_metrics(test_true, test_uncorrected)
    test_corrected_metrics = radiation_metrics(test_true, test_corrected)
    rmse_improvement = percent_improvement(test_baseline["rmse"], test_corrected_metrics["rmse"])
    mae_improvement = percent_improvement(test_baseline["mae"], test_corrected_metrics["mae"])

    station_dir.mkdir(parents=True, exist_ok=True)
    temporary_model = station_dir / "model.tmp.json"
    booster.save_model(temporary_model)
    os.replace(temporary_model, station_dir / "model.json")
    gain_table = feature_gain_table(booster, model_features)
    atomic_csv(station_dir / "feature_gain.csv", gain_table)

    predictions = test_frame[["ObservationPeriod", "station", "date"]].copy()
    predictions.insert(0, "case", case)
    predictions.insert(1, "held_out_station", station)
    predictions[f"observed_{case}"] = test_true
    predictions[f"CAMS_{case}"] = test_uncorrected
    predictions[f"predicted_correction_{case}"] = test_correction
    predictions[f"corrected_{case}"] = test_corrected
    predictions["uncorrected_error"] = test_uncorrected - test_true
    predictions["corrected_error"] = test_corrected - test_true
    atomic_csv(predictions_path, predictions)
    make_publication_figure(
        case,
        test_true,
        test_uncorrected,
        test_corrected,
        station_dir / "figures",
        grid_module,
    )

    metrics = {
        "case": case,
        "held_out_station": station,
        "created_at": utc_now(),
        "candidate_id": str(best["candidate_id"]),
        "num_boost_rounds": num_boost_rounds,
        "hyperparameters": {name: best[name] for name in SEARCH_PARAMETER_NAMES},
        "model_features": model_features,
        "seed": args.seed,
        "training_scope": "all usable observations from every station except the held-out station",
        "test_scope": "all usable observations from the held-out station",
        "training_stations": training_station_count,
        "training_rows": int(dtrain.num_row()),
        "test_rows": int(dtest.num_row()),
        "fit_seconds": fit_seconds,
        "training_baseline_metrics": train_baseline,
        "training_corrected_metrics": train_corrected_metrics,
        "test_baseline_metrics": test_baseline,
        "test_corrected_metrics": test_corrected_metrics,
        "test_rmse_improvement_percent": rmse_improvement,
        "test_mae_improvement_percent": mae_improvement,
        "feature_gain_percent": dict(zip(gain_table["feature"], gain_table["gain_percent"])),
        "loso_design": "non-nested",
        "selection_caveat": (
            "Hyperparameters and boosting rounds were frozen from the preceding "
            "station-date grid search, which used data from all stations. No LOSO "
            "fold result was used for tuning."
        ),
    }
    atomic_json(metrics_path, metrics)
    logger.info(
        "%s/%s: RMSE %.3f -> %.3f W/m² (%.2f%%); MAE %.3f -> %.3f W/m²",
        case,
        station,
        test_baseline["rmse"],
        test_corrected_metrics["rmse"],
        rmse_improvement,
        test_baseline["mae"],
        test_corrected_metrics["mae"],
    )
    return loso_summary_row(metrics), predictions


def pooled_loso_summary(
    case: str,
    predictions: pd.DataFrame,
    stations: list[str],
    best: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    true = predictions[f"observed_{case}"].to_numpy(dtype=np.float64)
    uncorrected = predictions[f"CAMS_{case}"].to_numpy(dtype=np.float64)
    corrected = predictions[f"corrected_{case}"].to_numpy(dtype=np.float64)
    baseline = radiation_metrics(true, uncorrected)
    corrected_metrics = radiation_metrics(true, corrected)
    summary = {
        "case": case,
        "created_at": utc_now(),
        "candidate_id": str(best["candidate_id"]),
        "num_boost_rounds": int(best["best_num_boost_rounds"]),
        "stations_evaluated": len(stations),
        "station_ids": stations,
        "test_rows": len(predictions),
        "baseline_metrics": baseline,
        "corrected_metrics": corrected_metrics,
        "rmse_improvement_percent": percent_improvement(baseline["rmse"], corrected_metrics["rmse"]),
        "mae_improvement_percent": percent_improvement(baseline["mae"], corrected_metrics["mae"]),
        "loso_design": "non-nested",
        "selection_caveat": (
            "Hyperparameters and boosting rounds were frozen from the preceding "
            "station-date grid search, which used data from all stations."
        ),
    }
    return summary, true, uncorrected, corrected


def run_loso_case(
    case: str,
    args: argparse.Namespace,
    grid_module,
    logger: logging.Logger,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    best, _ = validate_search_case(args.search_results, case)
    frame = grid_module.load_case(args.data_root, case, logger)
    available_stations = sorted(frame["station"].astype(str).unique().tolist())
    if args.stations is None:
        selected_stations = available_stations
    else:
        missing = sorted(set(args.stations) - set(available_stations))
        if missing:
            raise ValueError(f"{case}: requested station(s) not present after filtering: {missing}")
        selected_stations = [station for station in available_stations if station in args.stations]
    logger.info(
        "%s: running %d leave-one-station-out folds with candidate %s and %d rounds",
        case,
        len(selected_stations),
        best["candidate_id"],
        int(best["best_num_boost_rounds"]),
    )
    all_features, correction_target, model_features = prepare_model_data(frame, case, grid_module)
    station_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []
    case_dir = args.output_dir / case
    for fold_number, station in enumerate(selected_stations, start=1):
        logger.info("%s: station fold %d/%d (%s)", case, fold_number, len(selected_stations), station)
        row, predictions = fit_loso_station(
            case,
            station,
            frame,
            all_features,
            correction_target,
            model_features,
            best,
            args,
            grid_module,
            logger,
        )
        station_rows.append(row)
        prediction_parts.append(predictions)
        atomic_csv(case_dir / "station_metrics.csv", pd.DataFrame(station_rows))

    pooled_predictions = pd.concat(prediction_parts, ignore_index=True)
    if pooled_predictions.duplicated(["ObservationPeriod", "station"]).any():
        raise RuntimeError(f"{case}: duplicate observations found across LOSO folds")
    atomic_csv(case_dir / "pooled_station_predictions.csv", pooled_predictions)
    pooled, true, uncorrected, corrected = pooled_loso_summary(
        case, pooled_predictions, selected_stations, best
    )
    atomic_json(case_dir / "pooled_metrics.json", pooled)
    make_publication_figure(
        case,
        true,
        uncorrected,
        corrected,
        args.output_dir / "figures",
        grid_module,
    )
    return station_rows, pooled


def write_loso_summary(
    output_dir: Path,
    station_rows: list[dict[str, Any]],
    case_rows: list[dict[str, Any]],
) -> None:
    station_frame = pd.DataFrame(station_rows)
    case_frame = pd.DataFrame(
        [
            {
                "case": row["case"],
                "candidate_id": row["candidate_id"],
                "num_boost_rounds": row["num_boost_rounds"],
                "stations_evaluated": row["stations_evaluated"],
                "test_rows": row["test_rows"],
                "baseline_mean_bias_W_m2": row["baseline_metrics"]["mean_bias"],
                "corrected_mean_bias_W_m2": row["corrected_metrics"]["mean_bias"],
                "baseline_MAE_W_m2": row["baseline_metrics"]["mae"],
                "corrected_MAE_W_m2": row["corrected_metrics"]["mae"],
                "MAE_improvement_percent": row["mae_improvement_percent"],
                "baseline_RMSE_W_m2": row["baseline_metrics"]["rmse"],
                "corrected_RMSE_W_m2": row["corrected_metrics"]["rmse"],
                "RMSE_improvement_percent": row["rmse_improvement_percent"],
                "baseline_R2": row["baseline_metrics"]["r2"],
                "corrected_R2": row["corrected_metrics"]["r2"],
            }
            for row in case_rows
        ]
    )
    atomic_csv(output_dir / "overall_station_metrics.csv", station_frame)
    atomic_csv(output_dir / "overall_case_metrics.csv", case_frame)
    lines = [
        "Leave-one-station-out XGBoost evaluation",
        "=" * 43,
        "",
        "Each fold trains on all usable observations from every other station",
        "and evaluates on all usable observations from the omitted station.",
        "Hyperparameters and boosting rounds are frozen from the preceding",
        "station-date grid search. This is therefore a non-nested LOSO analysis.",
        "No LOSO result is used for model selection.",
        "",
    ]
    for row in case_rows:
        lines.extend(
            [
                f"{row['case']} ({row['stations_evaluated']} stations; {row['test_rows']:,} pooled predictions):",
                f"  Candidate: {row['candidate_id']}; rounds: {row['num_boost_rounds']}",
                f"  Pooled RMSE: {row['baseline_metrics']['rmse']:.3f} -> "
                f"{row['corrected_metrics']['rmse']:.3f} W/m² "
                f"({row['rmse_improvement_percent']:.2f}% improvement)",
                f"  Pooled MAE: {row['baseline_metrics']['mae']:.3f} -> "
                f"{row['corrected_metrics']['mae']:.3f} W/m² "
                f"({row['mae_improvement_percent']:.2f}% improvement)",
                f"  Pooled mean bias: {row['baseline_metrics']['mean_bias']:.3f} -> "
                f"{row['corrected_metrics']['mean_bias']:.3f} W/m²",
                f"  Pooled R²: {row['baseline_metrics']['r2']:.4f} -> "
                f"{row['corrected_metrics']['r2']:.4f}",
                "",
            ]
        )
    (output_dir / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


def bootstrap_radiation_metrics(true: np.ndarray, uncorrected: np.ndarray, corrected: np.ndarray, num_samples: int = 5000):
    baseline_metrics_bootstrap = []
    corrected_metrics_bootstrap = []
    n = len(true)
    for _ in range(num_samples):
        indices = np.random.randint(0, n, n)
        baseline_metrics_bootstrap.append(radiation_metrics(true[indices], uncorrected[indices]))
        corrected_metrics_bootstrap.append(radiation_metrics(true[indices], corrected[indices]))
    return baseline_metrics_bootstrap, corrected_metrics_bootstrap


def make_publication_figure(
    case: str,
    true: np.ndarray,
    uncorrected: np.ndarray,
    corrected: np.ndarray,
    figures_dir: Path,
    grid_module,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    maximum = float(grid_module.PLOT_MAX[case])
    baseline_metrics = radiation_metrics(true, uncorrected)
    corrected_metrics = radiation_metrics(true, corrected)
    # 5000 bootstrap samples to compute 95% percentile intervals for the metrics
    baseline_metrics_bootstrap, corrected_metrics_bootstrap = bootstrap_radiation_metrics(true, uncorrected, corrected, num_samples=5000)
    baseline_metrics_ci = {
        "median_bias": np.percentile([m["median_bias"] for m in baseline_metrics_bootstrap], [2.5, 97.5]),
        "mean_bias": np.percentile([m["mean_bias"] for m in baseline_metrics_bootstrap], [2.5, 97.5]),
        "mae": np.percentile([m["mae"] for m in baseline_metrics_bootstrap], [2.5, 97.5]),
        "rmse": np.percentile([m["rmse"] for m in baseline_metrics_bootstrap], [2.5, 97.5]),
        "r2": np.percentile([m["r2"] for m in baseline_metrics_bootstrap], [2.5, 97.5]),
    }
    corrected_metrics_ci = {
        "median_bias": np.percentile([m["median_bias"] for m in corrected_metrics_bootstrap], [2.5, 97.5]),
        "mean_bias": np.percentile([m["mean_bias"] for m in corrected_metrics_bootstrap], [2.5, 97.5]),
        "mae": np.percentile([m["mae"] for m in corrected_metrics_bootstrap], [2.5, 97.5]),
        "rmse": np.percentile([m["rmse"] for m in corrected_metrics_bootstrap], [2.5, 97.5]),
        "r2": np.percentile([m["r2"] for m in corrected_metrics_bootstrap], [2.5, 97.5]),
    }

    relative_rmse_change_95pct_intval = np.percentile(
        [percent_improvement(baseline_metrics_bootstrap[i]["rmse"], corrected_metrics_bootstrap[i]["rmse"]) for i in range(5000)],
        [2.5, 97.5],
    )
    print(f"Relative RMSE change ({case}): {relative_rmse_change_95pct_intval[0]:.2f}% - {relative_rmse_change_95pct_intval[1]:.2f}%")

    # 8 x 4 inch figure, two panels.
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
    panels = [
        (axes[0], uncorrected, baseline_metrics, baseline_metrics_ci, f"CAMS {case} (W/m²)"),
        (axes[1], corrected, corrected_metrics, corrected_metrics_ci, f"Bias-corrected CAMS {case} (W/m²)"),
    ]
    for axis, predicted, metrics, metrics_ci, ylabel in panels:
        axis.hist2d(
            true,
            predicted,
            bins=[80, 120],
            range=[[0.0, maximum], [0.0, maximum]],
            cmap=grid_module.color_map(),
            norm=LogNorm(),
            alpha=0.80,
            zorder=5,
        )
        axis.plot([0.0, maximum], [0.0, maximum], "k-", linewidth=1.2, zorder=10)
        axis.grid(True, color="0.65", linewidth=0.6)
        axis.set_xlim(0.0, maximum)
        axis.set_ylim(0.0, maximum)
        axis.set_xlabel(f"True {case} (W/m²)")
        axis.set_ylabel(ylabel)
        axis.text(
            0.02,
            0.98,
            f"Median bias: {metrics['median_bias']:.1f} ({metrics_ci['median_bias'][0]:.1f}, {metrics_ci['median_bias'][1]:.1f}) W/m²\n"
            f"Mean bias: {metrics['mean_bias']:.1f} ({metrics_ci['mean_bias'][0]:.1f}, {metrics_ci['mean_bias'][1]:.1f}) W/m²\n"
            f"MAE: {metrics['mae']:.1f} ({metrics_ci['mae'][0]:.1f}, {metrics_ci['mae'][1]:.1f}) W/m²\n"
            f"RMSE: {metrics['rmse']:.1f} ({metrics_ci['rmse'][0]:.1f}, {metrics_ci['rmse'][1]:.1f}) W/m²\n"
            f"R²: {metrics['r2']:.3f} ({metrics_ci['r2'][0]:.3f}, {metrics_ci['r2'][1]:.3f})",
            fontsize=8.5,
            bbox={"facecolor": "white", "alpha": 0.96, "edgecolor": "black"},
            ha="left",
            va="top",
            zorder=15,
            transform=axis.transAxes,
        )
    # Deliberately omit a title so separate case figures can be stacked.
    fig.tight_layout(pad=0.7)
    pdf_path = figures_dir / f"{case}_corrected.pdf"
    png_path = figures_dir / f"{case}_corrected.png"
    fig.savefig(pdf_path, bbox_inches=None)
    fig.savefig(png_path, dpi=200, bbox_inches=None)
    plt.close(fig)


def write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    atomic_csv(output_dir / "overall_final_metrics.csv", pd.DataFrame(rows))
    lines = [
        "Final XGBoost radiation error-correction models",
        "=" * 48,
        "",
        "Each model was refitted on training + validation data with frozen",
        "hyperparameters and boosting rounds, then evaluated on the held-out test set.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"{row['case']}:",
                f"  Candidate: {row['candidate_id']}",
                f"  Boosting rounds: {row['num_boost_rounds']}",
                f"  Training rows: {row['training_rows']:,}; test rows: {row['test_rows']:,}",
                f"  Test RMSE: {row['baseline_RMSE_W_m2']:.3f} -> {row['corrected_RMSE_W_m2']:.3f} W/m² "
                f"({row['RMSE_improvement_percent']:.2f}% improvement)",
                f"  Test MAE: {row['baseline_MAE_W_m2']:.3f} -> {row['corrected_MAE_W_m2']:.3f} W/m² "
                f"({row['MAE_improvement_percent']:.2f}% improvement)",
                f"  Test mean bias: {row['baseline_mean_bias_W_m2']:.3f} -> "
                f"{row['corrected_mean_bias_W_m2']:.3f} W/m²",
                f"  Test R²: {row['baseline_R2']:.4f} -> {row['corrected_R2']:.4f}",
                "",
            ]
        )
    (output_dir / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    logger = setup_logging(args.output_dir, args.leave_one_station_out)
    grid_module = load_grid_module(args.data_root)
    mode = "leave_one_station_out" if args.leave_one_station_out else "final_test"
    logger.info("Data root: %s", args.data_root)
    logger.info("Grid-search results: %s", args.search_results)
    logger.info("Final output directory: %s", args.output_dir)
    logger.info("Evaluation mode: %s", mode)
    logger.info("Python: %s", sys.executable)
    logger.info("XGBoost %s; using %d CPU threads", xgb.__version__, args.nthread)
    if args.leave_one_station_out:
        logger.warning(
            "LOSO is non-nested: hyperparameters and boosting rounds were selected "
            "once by the preceding station-date grid search using all stations."
        )
    sleep_process = prevent_macos_sleep(logger, args.allow_sleep)
    started = time.perf_counter()
    np.random.seed(args.seed)
    try:
        if args.leave_one_station_out:
            station_rows: list[dict[str, Any]] = []
            case_rows: list[dict[str, Any]] = []
            for case in args.cases:
                this_station_rows, this_case_row = run_loso_case(
                    case, args, grid_module, logger
                )
                station_rows.extend(this_station_rows)
                case_rows.append(this_case_row)
                write_loso_summary(args.output_dir, station_rows, case_rows)
            method = (
                "Non-nested leave-one-station-out evaluation: parameters and rounds "
                "frozen from the preceding grid search; each fold trained on all rows "
                "from all other stations and evaluated on all rows from the omitted station."
            )
        else:
            rows: list[dict[str, Any]] = []
            for case in args.cases:
                rows.append(fit_final_case(case, args, grid_module, logger))
                write_summary(args.output_dir, rows)
            method = (
                "Hyperparameters and boosting rounds frozen from grid search; "
                "refit on train + validation; final evaluation on test."
            )
        atomic_json(
            args.output_dir / "run_manifest.json",
            {
                "created_at": utc_now(),
                "command": sys.argv,
                "data_root": args.data_root,
                "search_results": args.search_results,
                "output_dir": args.output_dir,
                "cases": args.cases,
                "mode": mode,
                "requested_stations": args.stations,
                "seed": args.seed,
                "nthread": args.nthread,
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "xgboost_version": xgb.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
                "method": method,
                "loso_is_nested": False if args.leave_one_station_out else None,
            },
        )
        logger.info("All requested models completed in %.1f seconds", time.perf_counter() - started)
        return 0
    finally:
        if sleep_process is not None and sleep_process.poll() is None:
            sleep_process.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
