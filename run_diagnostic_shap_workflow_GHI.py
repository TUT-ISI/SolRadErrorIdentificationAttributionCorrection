#!/usr/bin/env python3
"""Example workflow: data preparation, diagnostic XGBoost model and SHAP (GHI).

Steps:
1. Build the collocated CRS dataset (CAMS_dataset.h5) from the CRS time-series
   CSV files and the ground-observation dataset (obs_dataset.h5) from the
   level-1 station NetCDF files (15 min centred rolling mean).
2. Apply the station-quality and data-quality filters.
3. Fit an in-sample diagnostic XGBoost model of the residual GHI_obs - GHI_CRS
   (100,000 sampled rows, max_depth=15, colsample_bytree=0.3, learning_rate=0.9,
   100 rounds) once for each random seed ID 01..10.
4. Compute TreeSHAP values and make beeswarm, dependence and waterfall plots.

The MSG pixel geometry and acquisition-time helpers are included below.  The
raw data are not included.


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

import logging
import os
from glob import glob
from types import SimpleNamespace

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xarray as xr
import xgboost
from matplotlib.axis import XAxis
from matplotlib.colors import LinearSegmentedColormap, LogNorm


# Station classification for 2021 in the SEVIRI field of view:
# (quality GHI, quality BNI, station type)
# Quality: v = very good, g = good (use with caution), m = maybe (not recommended),
# b = bad (do not use).  Type: C = continental, D = desert, P = polar,
# S = sub-pixel (coastal/island, < 10 km to coast), M = mountainous (> 1500 m).
sts_classes = {
    "BSRN": {
        "BUD": ("g", "g", "C"),
        "CAB": ("v", "v", "C"),
        "CNR": ("v", "v", "C"),
        "FLO": ("g", "g", "S"),
        "GOB": ("v", "v", "D"),
        "GVN": ("g", "g", "P"),
        "INO": ("g", "g", "C"),
        "IZA": ("g", "g", "S"),
        "LIN": ("v", "v", "C"),
        "NYA": ("g", "g", "P"),
        "PAL": ("g", "g", "C"),
        "PAY": ("v", "v", "C"),
        "RUN": ("g", "g", "S"),
        "SON": ("g", "g", "M"),
        "SPO": ("m", "m", "P"),
        "SYO": ("g", "g", "P"),
        "TAM": ("b", "-", "D"),
    },
    "enerMENA": {
        "CAI": ("m", "b", "D"),
        "ERF": ("v", "v", "D"),
        "FKI": ("b", "b", "D"),
        "JOR": ("g", "g", "D"),
        "MIS": ("m", "m", "D"),
        "OUJ": ("b", "b", "D"),
        "TN": ("v", "v", "D"),
        "ZAG": ("v", "v", "D"),
    },
    "SAURAN": {
        "CSIR": ("g", "g", "C"),
        "CUT": ("g", "g", "C"),
        "ELA": ("m", "m", "C"),
        "FRH": ("m", "m", "C"),
        "HLO": ("b", "b", "S"),
        "KZH": ("b", "b", "S"),
        "NMU": ("g", "g", "S"),
        "NUST": ("b", "b", "M"),
        "STA": ("m", "b", "S"),
        "SUN": ("g", "g", "C"),
        "UNV": ("m", "b", "C"),
        "UNZ": ("m", "b", "C"),
        "UPR": ("v", "v", "C"),
    }
}


# Columns of the CRS time-series CSV files (irradiations in Wh/m2 per minute,
# converted to W/m2 below; AOD at 550 nm; Snow_probability, CloudCoverage and
# CloudType are -1 when missing; CloudType 0 = no clouds, 5-8 = cloud classes).
columns = [
    "ObservationPeriod",
    "TOA",
    "ClearSkyGHI",
    "ClearSkyBHI",
    "ClearSkyDHI",
    "ClearSkyBNI",
    "GHI",
    "BHI",
    "DHI",
    "BNI",
    "Reliability",
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
    "alpha",
    "Snow_probability",
    "fiso",
    "fvol",
    "fgeo",
    "albedo",
    "CloudOpticalDepth",
    "CloudCoverage",
    "CloudType",
    "GHInocorr",
    "BHInocorr",
    "DHInocorr",
    "BNInocorr",
]

model_inputs = [
    "TOA",
    "GHI",
    "BHI",
    "DHI",
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


# MSG (Meteosat Second Generation) pixel geometry and acquisition-time helpers.
# Based on code by Benoit Gschwind (MINES Paris - PSL); geostationary projection
# after the LRIT/HRIT Global Specification, Issue 2.6, August 1999.

def normalized_geostationary_projection_xy_to_lonlat(x, y, lon_sat, h=0.0):
    """
    Approximate conversion of normalised geostationary x, y (radians, [1]) to
    longitude and latitude (degrees), with an altitude parameter h (km).
    """
    r_pol = 6356.5838 + h  # polar radius in km [1]
    r_eq = 6378.1690 + h  # equatorial radius in km [1]
    hsat = 42164.0  # satellite distance from the centre of the Earth in km [1]

    lon_sat = np.radians(lon_sat)

    sin_y = np.sin(y)
    cos_y = np.cos(y)
    sin_x = np.sin(x)
    cos_x = np.cos(x)

    k0 = (r_eq * r_eq) / (r_pol * r_pol)
    k1 = hsat * cos_x * cos_y
    k2 = cos_y * cos_y + k0 * sin_y * sin_y

    sd = np.sqrt(k1 * k1 - k2 * 1737121856)
    sn = (k1 - sd) / k2

    s1 = hsat - sn * cos_x * cos_y
    s2 = sn * sin_x * cos_y
    s3 = -sn * sin_y

    sxy = np.sqrt(s1 * s1 + s2 * s2)

    lon = np.arctan(s2 / s1) + lon_sat
    lat = np.arctan((k0 * s3) / sxy)

    return np.degrees(lon), np.degrees(lat)


def normalized_geostationary_projection_lonlat_to_xy(lon, lat, lon_sat, h=0.0):
    """
    Approximate conversion of longitude and latitude (degrees) to normalised
    geostationary x, y (radians, [1]), with an altitude parameter h (km).
    """
    r_pol = 6356.5838 + h  # polar radius in km [1]
    r_eq = 6378.1690 + h  # equatorial radius in km [1]
    h_sat = 42164.0  # satellite distance from the centre of the Earth in km [1]

    lat = np.radians(lat)
    lon = np.radians(lon)
    lon_sat = np.radians(lon_sat)

    k1 = (r_pol * r_pol) / (r_eq * r_eq)
    k2 = (r_eq * r_eq - r_pol * r_pol) / (r_eq * r_eq)

    phi_e = np.arctan(np.tan(lat) * k1)
    cos_phi_e = np.cos(phi_e)

    re = r_pol / np.sqrt(1.0 - k2 * cos_phi_e * cos_phi_e)

    r1 = h_sat - re * cos_phi_e * np.cos(lon - lon_sat)
    r2 = -re * cos_phi_e * np.sin(lon - lon_sat)
    r3 = re * np.sin(phi_e)

    x = np.arctan(-r2 / r1)
    y = np.arcsin(-r3 / np.sqrt(r1 * r1 + r2 * r2 + r3 * r3))
    return x, y


def lonlat_to_msg_pixel(lon, lat, h=0.0):
    """MSG pixel coordinates (1 to 3712) of a longitude/latitude."""
    x, y = normalized_geostationary_projection_lonlat_to_xy(lon, lat, 0.0, h)
    fac = -781648343.404
    delta_x = 65536 / fac
    delta_y = 65536 / fac
    return x / delta_x + 1856.0, y / delta_y + 1856.0


def msg_pixel_to_lonlat(px, py, h=0.0):
    """Longitude/latitude of MSG pixel coordinates (1 to 3712)."""
    fac = -781648343.404
    delta_x = 65536 / fac
    delta_y = 65536 / fac
    x = (px - 1856.0) * delta_x
    y = (py - 1856.0) * delta_y
    return normalized_geostationary_projection_xy_to_lonlat(x, y, 0.0, h)


def hc3_offset(lig):
    """
    Scan-time offset (fraction of an hour) of line lig (counting from 1) in the
    Meteosat image.
    """
    I3_NBLIG = 3712
    OFFSET_LINE = 19
    NBSTEP = (I3_NBLIG + 2.0 * OFFSET_LINE) / 3.0
    delta_t = (I3_NBLIG - (lig - 1) + OFFSET_LINE) / 3.0
    return delta_t * (12.5 / NBSTEP / 60.0)


def lonlat_to_hc3_pixel(lon, lat):
    px, py = lonlat_to_msg_pixel(lon, lat)
    return 3713 - px, 3713 - py


def hc3_pixel_to_lonlat(qx, qy):
    return msg_pixel_to_lonlat(3713 - qx, 3713 - qy)


def get_pxl_data_new(lon, lat):
    """MSG pixel (indices, centre coordinates and scan-time offset in days) of a location."""
    pxl = SimpleNamespace()
    pxl.px, pxl.py = lonlat_to_hc3_pixel(lon, lat)
    pxl.px = int(pxl.px + .5) - 1
    pxl.py = int(pxl.py + .5) - 1
    pxl.lon, pxl.lat = hc3_pixel_to_lonlat(pxl.px, pxl.py)
    pxl.hdelta = hc3_offset(pxl.py + 1) / 24.0
    return pxl


def get_pxl_time(pxl, datetime_start, length):
    """Acquisition times of the pixel: 15 min steps from datetime_start plus the scan offset."""
    return np.datetime64(datetime_start) \
        + np.timedelta64(int(pxl.hdelta * 24 * 60 * 60 * 1000000), "us") \
        + np.arange(length, dtype="u4") * np.timedelta64(15, "m")


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


cmap_2d_hist = color_map()


def prepare_cams_dataset():
    np.random.seed(0)
    df_combined = None

    obsfiles = sorted(glob("CAMEOglobaldata/dataset2_v1/crs/*.csv"))
    for obsfile in obsfiles:
        filename = os.path.split(obsfile)[-1]
        if os.stat(obsfile).st_size == 0:
            logging.info(f"Skipping empty file: {filename}")
            continue
        logging.info(f"Preparing data: {filename}")

        lat, lon, elevation, CAMS_elevation = None, None, None, None
        with open(obsfile, "rt") as f:
            for ii in range(20):
                ll = f.readline()
                if ll.find("# Latitude (positive North, ISO 19115): ") >= 0:
                    lat = float(ll.removeprefix("# Latitude (positive North, ISO 19115): ").removesuffix("\n"))
                if ll.find("# Longitude (positive East, ISO 19115): ") >= 0:
                    lon = float(ll.removeprefix("# Longitude (positive East, ISO 19115): ").removesuffix("\n"))
                if ll.find("# Altitude (m): ") >= 0:
                    elevation = float(ll.removeprefix("# Altitude (m): ").removesuffix("\n"))
                if ll.find("# Elevation of CAMS cell (m): ") >= 0:
                    CAMS_elevation = float(ll.removeprefix("# Elevation of CAMS cell (m): ").removesuffix("\n"))

        df = pd.read_csv(obsfile, sep=";", index_col=False, comment="#", header=None, names=columns)
        df["ObservationPeriod"] = df["ObservationPeriod"].str.split("/")
        df["ObservationPeriod"] = pd.to_datetime(df["ObservationPeriod"].apply(lambda x: x[0]))
        df["TOA"] = df["TOA"] * 60.
        df["ClearSkyGHI"] = df["ClearSkyGHI"] * 60.
        df["ClearSkyBHI"] = df["ClearSkyBHI"] * 60.
        df["ClearSkyDHI"] = df["ClearSkyDHI"] * 60.
        df["ClearSkyBNI"] = df["ClearSkyBNI"] * 60.
        df["GHI"] = df["GHI"] * 60.
        df["BHI"] = df["BHI"] * 60.
        df["DHI"] = df["DHI"] * 60.
        df["BNI"] = df["BNI"] * 60.

        pxl = get_pxl_data_new(lon, lat)
        timeMSG = np.array(get_pxl_time(pxl, df["ObservationPeriod"].iloc[0], int(np.ceil((df["ObservationPeriod"].iloc[-1] - df["ObservationPeriod"].iloc[0]) / np.timedelta64(15, "m")))), dtype="datetime64[m]")
        df.loc[df.GHI < 0, "GHI"] = np.nan
        df.loc[df.DHI < 0, "DHI"] = np.nan
        df.loc[df.BNI < 0, "BNI"] = np.nan
        additional_inputs = ["ObservationPeriod", "TOA", "ClearSkyGHI", "ClearSkyBHI", "ClearSkyDHI", "ClearSkyBNI", "GHI", "BHI", "DHI", "BNI"]
        df = df[list(dict.fromkeys(model_inputs + additional_inputs))]
        df = df.dropna().reset_index()
        # keep only MSG acquisition times
        df = pd.DataFrame(df[np.isin(df["ObservationPeriod"], timeMSG)])
        df["lat"] = lat
        df["lon"] = lon
        df["elevation"] = elevation
        df["CAMSelevation"] = CAMS_elevation
        df["station"] = os.path.split(obsfile)[-1].removesuffix(".csv")

        if df_combined is None:
            df_combined = df.reset_index()
        else:
            df_combined = pd.concat((df_combined, df.reset_index()), ignore_index=True)

    return df_combined


def prepare_obs_dataset(stations, CAMSdf):
    df_combined = None
    for station in stations:
        logging.info(f"Loading observations: {station}")
        datafile = os.path.join("CAMEOglobaldata", "dataset2_v1", "gnd", f'{station.replace("_", "-")}.l1.nc')
        if not os.path.isfile(datafile):
            logging.error(f"L1 FILE NOT FOUND: {datafile}")
        print("")
        stationOBSdf = xr.open_dataset(datafile).to_pandas()[["GHI", "DHI", "BNI"]]
        stationOBSdf = stationOBSdf.rolling("15min", center=True).mean().dropna()
        stationCAMSdf = CAMSdf[CAMSdf["station"] == station]
        stationOBSdf = stationOBSdf[np.isin(stationOBSdf.index, stationCAMSdf["ObservationPeriod"])]
        stationOBSdf["station"] = station
        stationOBSdf = stationOBSdf.reset_index().rename({"time": "ObservationPeriod"}, axis=1)

        if df_combined is None:
            df_combined = stationOBSdf
        else:
            df_combined = pd.concat((df_combined, stationOBSdf), ignore_index=True)

    return df_combined


def preprocess_data(CAMSdf, OBSdf):
    logging.info("Preprocessing data")
    CAMSdf = CAMSdf.set_index(["ObservationPeriod", "station"])
    OBSdf = OBSdf.set_index(["ObservationPeriod", "station"])

    df = pd.concat([CAMSdf, OBSdf.add_suffix("_obs", axis=1)], axis=1).reset_index()

    df = df[(df["sza"] < 80.) & (df["GHI"] > 0.0) & (df["DHI"] > 0.0) & (df["GHI_obs"] > 0.0) & (df["DHI_obs"] > 0.0) & (df["TOA"] > 0.0) & (df["Snow_probability"] >= 0.0) & (~np.isnan(df["CloudOpticalDepth"])) & (df["CloudType"] >= 0.0) & (df["CloudCoverage"] >= 0.0)]
    df = df.dropna(axis="rows", subset=["GHI", "GHI_obs"])

    inputs = df[model_inputs].fillna(-1)
    outputs = np.array(df["GHI_obs"] - df["GHI"])
    logging.info("Preprocessing done")
    return df, inputs, outputs


def plot_initial_data(df, outputs, ID):
    logging.info("Plotting initial data")
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])
    ax.grid(linestyle="--", c="#404040", linewidth=0.5, alpha=0.3, zorder=0)
    ax.hist(outputs, bins=101, range=(-300, 300), zorder=1)
    ax.set_xlim([-300, 300])
    ax.set_xlabel("GHI observed - GHI CAMS (W / m²)")
    ax.set_ylabel("N")
    plt.savefig(f"figs/{ID}/GHI_error_histogram_all_data.pdf")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.hist2d(df["GHI_obs"], df["GHI"], bins=(80, 120), range=[[0, 1200], [0, 1200]], cmap=cmap_2d_hist, norm=LogNorm(), alpha=0.8, zorder=5)
    ax.plot([0, 1200], [0, 1200], "k-", zorder=10)
    ax.grid(True)
    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 1200])
    ax.set_title("", fontsize=15, c="k")
    ax.set_xlabel("Observed GHI (W/m²)")
    ax.set_ylabel("CAMS GHI (W/m²)")
    plt.savefig(f"figs/{ID}/GHI_comparison.pdf")
    plt.close("all")

    logging.info("Plotting initial data done")


def train_xgboost(inputs, outputs, ID):
    Ntotalsamples = len(outputs)
    Ntestsamples = 100000
    np.random.seed(1)
    samplesTest = np.random.randint(Ntotalsamples, size=Ntestsamples)
    samplesTrain = np.array(samplesTest)

    X_train, X_test, y_train, y_test = inputs.iloc[samplesTrain], inputs.iloc[samplesTest], outputs[samplesTrain], outputs[samplesTest]
    RMSECAMS = np.sqrt(np.mean((outputs)**2))
    logging.info(f"RMSE of CAMS data: {RMSECAMS}")

    np.random.seed(int(ID))

    logging.info("Training XGboost model")
    Xd = xgboost.DMatrix(X_train, label=y_train)
    bst = xgboost.train(params={
        "max_depth": 15,
        "colsample_bytree": 0.3,
        "learning_rate": 0.9,
        "objective": "reg:squarederror",
        "nthread": 8,
        "seed": int(ID)},
        dtrain=Xd,
        num_boost_round=100)
    y_train_pred = bst.predict(Xd)
    y_test_pred = bst.predict(xgboost.DMatrix(X_test, label=y_test))
    RMSEtrain, RMSEtest = np.sqrt(np.mean((y_train_pred - y_train)**2)), np.sqrt(np.mean((y_test_pred - y_test)**2))
    logging.info(f"RESULT {RMSEtrain} - {RMSEtest}")
    logging.info("Training done.")
    return bst, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred


def visualize_xgboost_performance(y_train, y_train_pred, y_test, y_test_pred, ID):
    logging.info("Plotting performance figures.")
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.hist2d(y_train, y_train_pred, bins=[80, 120], range=[[-200, 200], [-200, 200]], cmap=cmap_2d_hist, norm=LogNorm(), alpha=0.8, zorder=5)
    ax.plot([-200, 200], [-200, 200], "k-", zorder=10)
    ax.grid(True)
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_title("Training data", fontsize=15, c="k")
    ax.set_xlabel("True error in GHI (W/m²)")
    ax.set_ylabel("Predicted error in GHI (W/m²)")
    plt.savefig(f"figs/{ID}/XGB_train.pdf")
    plt.close("all")

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.hist2d(y_test, y_test_pred, bins=(80, 120), range=[[-200, 200], [-200, 200]], cmap=cmap_2d_hist, norm=LogNorm(), alpha=0.8, zorder=5)
    ax.plot([-200, 200], [-200, 200], "k-", zorder=10)
    ax.grid(True)
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_title("Test data", fontsize=15, c="k")
    ax.set_xlabel("True error in GHI (W/m²)")
    ax.set_ylabel("Predicted error in GHI (W/m²)")
    plt.savefig(f"figs/{ID}/XGB_test.pdf")
    plt.close("all")

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])
    ax.grid(linestyle="--", c="#404040", linewidth=0.5, alpha=0.3, zorder=0)
    ax.hist(y_train - y_train_pred, bins=101, range=(-200, 200), zorder=1)
    ax.set_xlim([-200, 200])
    ax.set_xlabel("GHI observed - GHI CAMS (W / m²)")
    ax.set_ylabel("N")
    plt.savefig(f"figs/{ID}/GHI_error_histogram_train.pdf")
    plt.close()

    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_axes([0.16, 0.16, 0.8, 0.8])
    ax.grid(linestyle="--", c="#404040", linewidth=0.5, alpha=0.3, zorder=0)
    ax.hist(y_test - y_test_pred, bins=101, range=(-200, 200), zorder=1)
    ax.set_xlim([-200, 200])
    ax.set_xlabel("GHI observed - GHI CAMS (W / m²)")
    ax.set_ylabel("N")
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/{ID}/GHI_error_histogram_test.pdf")
    plt.close()

    logging.info("Performance figures done.")


def compute_shap_values(bst, X_train):
    logging.info("Running SHAP")
    explainer = shap.Explainer(bst, X_train)
    shap_values = explainer(X_train)
    logging.info("SHAP done!")
    return explainer, shap_values


def plot_shap_values(shap_values, ID):
    logging.info("Visualizing SHAP results.")
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes([0.35, 0.12, 0.65, 0.85])
    shap.plots.beeswarm(shap_values, max_display=99, show=False, alpha=0.65)
    ax.set_xlim([-150, 150])
    plt.xlabel("SHAP value: impact on $GHI_{obs} - GHI_{CAMS}$", fontsize=13)
    plt.savefig(f"figs/{ID}/GHI_SHAP01.pdf", dpi=200)
    plt.close()

    for variablename in model_inputs:
        fig = plt.figure(figsize=(6, 4), dpi=200)
        ax = fig.add_axes([0.17, 0.12, 0.75, 0.85])
        shap.plots.scatter(shap_values[:, variablename], color="#384f85", ax=ax, dot_size=8, alpha=0.5, show=False)
        ax.axhline(0, zorder=20, linestyle="-", color="k", linewidth=0.5)
        plt.savefig(f"figs/{ID}/GHI_SHAP_variable_{variablename}.pdf")
        plt.close()

    logging.info("Visualizing SHAP results done.")


def plot_shap_waterfall(shap_values, y_train_pred, ID):
    if len(glob(f"figs/{ID}/GHI_SHAP_waterfall_*.pdf")) > 0:
        return None
    casesbypercentile = np.percentile(y_train_pred, [0.1, 0.2, 0.3, 0.4, 0.5, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 99.5, 99.6, 99.7, 99.8, 99.9])
    cases = [int(np.argwhere(y_train_pred == y_train_pred[np.argwhere(y_train_pred >= c)].min())[0]) for c in casesbypercentile]

    for ii, case in enumerate(cases, 1):
        fig = plt.figure(figsize=(3, 4), dpi=200)
        fig.add_axes([0.35, 0.15, 0.5, 0.75])
        shap.plots.waterfall(shap_values[case], show=False)
        matches = fig.findobj(lambda artist: isinstance(artist, XAxis))
        for match in matches:
            labels = match.get_ticklabels()
            new_labeltexts = []
            for label in labels:
                labeltext = label.get_text()
                if labeltext.find("f(") >= 0:
                    labeltext = labeltext.replace("f(X)", "GHI_{obs} - GHI_{CAMS}").replace("f(x)", "GHI_{obs} - GHI_{CAMS}")
                new_labeltexts.append(labeltext)
            if (len(new_labeltexts) == 2 and new_labeltexts[0][:1] == "\n") or ((len(new_labeltexts) == 2 and new_labeltexts[0][:1] == "$")):
                match.set_ticklabels([new_labeltexts[0] + new_labeltexts[1], ""])
        figsize = fig.get_size_inches()
        figsize[0] = figsize[0] * 2
        fig.set_size_inches(figsize)
        plt.savefig(f"figs/{ID}/GHI_SHAP_waterfall_{ii:03d}_{case:05d}.pdf")
        plt.close()


def main(ID):
    for dirtocreate in [os.path.join("figs", ID), os.path.join("model", ID), "data"]:
        os.makedirs(dirtocreate, exist_ok=True)

    if not os.path.isfile(os.path.join("data", "CAMS_dataset.h5")):
        CAMSdf = prepare_cams_dataset()
        CAMSdf.to_hdf(os.path.join("data", "CAMS_dataset.h5"), key="df", mode="w")
    else:
        CAMSdf = pd.read_hdf(os.path.join("data", "CAMS_dataset.h5"))

    if not os.path.isfile(os.path.join("data", "obs_dataset.h5")):
        stations = list(set(CAMSdf["station"]))
        OBSdf = prepare_obs_dataset(stations, CAMSdf)
        OBSdf.to_hdf(os.path.join("data", "obs_dataset.h5"), key="df", mode="w")
    else:
        OBSdf = pd.read_hdf(os.path.join("data", "obs_dataset.h5"))

    # keep only stations of quality classes v and g
    stationsAccept = []
    for network in sts_classes:
        for station in sts_classes[network]:
            if bool(np.isin(sts_classes[network][station][0], ["v", "g"])):
                stationsAccept.append(f"{network}_{station}")

    CAMSdf = CAMSdf[CAMSdf["station"].isin(stationsAccept)]
    OBSdf = OBSdf[OBSdf["station"].isin(stationsAccept)]
    df, inputs, outputs = preprocess_data(CAMSdf, OBSdf)
    CAMSdf, OBSdf = None, None
    plot_initial_data(df, outputs, ID)

    if not os.path.isfile(f"model/{ID}/explainerData.joblib"):
        bst, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = train_xgboost(inputs, outputs, ID)
        visualize_xgboost_performance(y_train, y_train_pred, y_test, y_test_pred, ID)
        explainer, shap_values = compute_shap_values(bst, X_train)
        joblib.dump({"df": df, "explainer": explainer, "shap_values": shap_values, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "y_train_pred": y_train_pred, "y_test_pred": y_test_pred}, filename=f"model/{ID}/explainerData.joblib", compress=("bz2", 9))
        bst.save_model(f"model/{ID}/bst.json")
    else:
        explainerData = joblib.load(f"model/{ID}/explainerData.joblib")
        explainer = explainerData["explainer"]
        shap_values = explainerData["shap_values"]
        y_train_pred = explainerData["y_train_pred"]

    plot_shap_values(shap_values, ID)
    plot_shap_waterfall(shap_values, y_train_pred, ID)

    logging.info("All done!")


if __name__ == "__main__":
    os.chdir(os.path.split(__file__)[0])
    for ID in [f"{ii:02d}" for ii in range(1, 10 + 1)]:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO, format=f"ID:{ID} - %(asctime)s - %(levelname)s - %(message)s")
        mpl.use("Agg")
        main(ID)
