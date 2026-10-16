"""Обычные функции построения графиков для zgamma_analysis.py."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "results_simple" / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


COLORS = {1: "#2475c7", 0: "#db5835"}
NAMES = {1: "Сигнал", 0: "Совокупный фон"}
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.family": "DejaVu Sans",
                     "axes.grid": True, "grid.alpha": 0.22, "axes.spines.top": False,
                     "axes.spines.right": False})


def save_figure(figure, folder, name, tight=True):
    if tight:
        figure.tight_layout()
    figure.savefig(folder / name, bbox_inches="tight")
    plt.close(figure)


def input_distributions(table, features, units, folder):
    for start in range(0, len(features), 9):
        selected_features = features[start:start + 9]
        rows = (len(selected_features) + 2) // 3
        figure, axes = plt.subplots(rows, 3, figsize=(15, rows * 3.6), squeeze=False)
        for index, feature in enumerate(selected_features):
            axis = axes.flat[index]
            for label in [1, 0]:
                selected = table.loc[(table.feature == feature) & (table["class"] == label)]
                for variant, style, weight_name in [("parquet_abs", "-", "|w|"),
                                                     ("parquet_signed", "--", "w")]:
                    bins = selected.loc[selected.variant == variant].sort_values("bin")
                    edges = np.append(bins.left.to_numpy(), bins.right.iloc[-1])
                    axis.stairs(bins.normalized, edges, color=COLORS[label], linestyle=style,
                                label=NAMES[label] + ": " + weight_name)
            axis.set_title(feature)
            axis.set_xlabel(units[feature])
            axis.set_ylabel("Доля / бин")
            if feature.startswith("n_"):
                axis.set_xticks(bins.left.to_numpy() + 0.5)
        for index in range(len(selected_features), rows * 3):
            axes.flat[index].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.03))
        page = start // 9 + 1
        figure.suptitle(f"Train: распределения с модулями и знаковыми весами, часть {page}", y=1.07)
        save_figure(figure, folder, f"input_distributions_{page}.png")


def learning_curves(histories, folder):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, name in zip(axes, ["XGBoost", "LightGBM"]):
        for trial in range(1, 5):
            values = histories[name + "_" + str(trial)]["validation"]["auc"]
            axis.plot(np.arange(1, len(values) + 1), values, label=f"Вариант {trial}")
        axis.set_title(name)
        axis.set_xlabel("Число итераций")
        axis.set_ylabel("Validation weighted AUC")
        axis.legend()
    save_figure(figure, folder, "validation_learning_curves.png")


def thresholds(scans, best, folder):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for axis, name in zip(axes, ["XGBoost", "LightGBM"]):
        scan = scans[name]
        valid = scan.ZA.notna()
        threshold = best[name]["threshold"]
        value = best[name]["validation_yields"]["ZA"]
        axis.plot(scan.loc[valid, "threshold"], scan.loc[valid, "ZA"])
        axis.scatter([threshold], [value], color="#db5835", zorder=3,
                     label=f"t={threshold:.5f}; ZA={value:.3f}")
        axis.set_title(name)
        axis.set_xlabel("Порог score")
        axis.set_ylabel("Validation ZA")
        axis.legend()
    save_figure(figure, folder, "validation_threshold_significance.png")


def roc_curves(table, metrics, folder):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, color in [("XGBoost", "#2475c7"), ("LightGBM", "#db5835")]:
        for split, style in [("validation", "--"), ("test", "-")]:
            curve = table.loc[(table.model == name) & (table.split == split)]
            auc = metrics.loc[(metrics.model == name) & (metrics.split == split), "weighted_auc"].iloc[0]
            for axis in axes:
                axis.plot(curve.FPR, curve.TPR, color=color, linestyle=style,
                          label=f"{name} {split}: AUC={auc:.5f}")
    for axis in axes:
        axis.set_xlabel("Эффективность фона (FPR, |w|)")
        axis.set_ylabel("Эффективность сигнала (TPR, |w|)")
        axis.set_ylim(0, 1.02)
        axis.legend(fontsize=9)
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle=":")
    axes[0].set_title("Полная ROC")
    axes[0].set_xlim(0, 1)
    axes[1].set_title("Область малого пропускания фона")
    axes[1].set_xscale("log")
    axes[1].set_xlim(1e-5, 0.2)
    save_figure(figure, folder, "weighted_roc.png")


def responses(table, ratios, folder):
    for name in ["XGBoost", "LightGBM"]:
        figure, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1, 1], "hspace": 0.12})
        for label, ratio_axis in [(1, axes[1]), (0, axes[2])]:
            selected = table.loc[(table.model == name) & (table["class"] == label)]
            train = selected.loc[selected.split == "train"].sort_values("bin")
            test = selected.loc[selected.split == "test"].sort_values("bin")
            edges = np.append(train.left.to_numpy(), train.right.iloc[-1])
            centers = (edges[:-1] + edges[1:]) / 2
            axes[0].stairs(train.normalized, edges, color=COLORS[label], fill=True,
                           alpha=0.15, label=NAMES[label] + ", train")
            axes[0].stairs(train.normalized, edges, color=COLORS[label])
            axes[0].errorbar(centers, test.normalized, yerr=test.error, color=COLORS[label],
                             fmt="o", markersize=3, capsize=2, label=NAMES[label] + ", test")
            ratio = ratios.loc[(ratios.model == name) & (ratios["class"] == label)].sort_values("bin")
            valid = ~ratio.zero_train_denominator
            ratio_axis.errorbar(centers[valid], ratio.loc[valid, "ratio"],
                                yerr=ratio.loc[valid, "full_error"], fmt="o", markersize=3,
                                color=COLORS[label], capsize=2)
            ratio_axis.axhline(1, color="gray", linestyle="--")
            if (~valid).any():
                ratio_axis.scatter(centers[~valid], np.full((~valid).sum(), 0.1),
                                   marker="x", color="black", label="Train=0")
                ratio_axis.legend(fontsize=8)
            ratio_axis.set_ylabel("Test/Train\n" + NAMES[label], fontsize=9)
        axes[0].set_title(name + ": отклик с исходными |w|")
        axes[0].set_ylabel("Доля / бин")
        axes[0].legend(ncol=2)
        axes[2].set_xlabel("Score (больше — ближе к сигналу)")
        axes[2].set_xlim(0, 1)
        figure.subplots_adjust(left=0.15, right=0.98, bottom=0.10, top=0.94)
        save_figure(figure, folder, name.lower() + "_train_test_response.png", tight=False)
