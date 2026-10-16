r"""Zgamma_EWK против десяти размеченных MC-фонов.

Запуск: .venv\Scripts\python.exe -X utf8 zgamma_analysis.py
Новые результаты: results_simple/. Старые results/ не изменяются.
Графики находятся в zgamma_plots.py, сверка — в compare_results.py.
Реальные неразмеченные данные этот сценарий не открывает.
"""

from pathlib import Path
import gc
import importlib.metadata
import json
import platform
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

import zgamma_plots as plots


# %% 1. Настройки. Порядок файлов и признаков совпадает со старым расчётом.
BASE = Path(__file__).resolve().parent
DATA = BASE / "Z(nunu)gamma+jj"
OUT = BASE / "results_simple"
SEED = 20261004
THREADS = 8
MAX_ROUNDS = 400
PATIENCE = 30
HOLDOUT_FRACTION = 0.15

SIGNAL = "Zgamma_EWK"
PROCESSES = [
    SIGNAL,
    "Zgamma_QCD",
    "Wgamma_QCD",
    "Wgamma_EWK",
    "top",
    "ttgamma",
    "gammajet",
    "Zj",
    "Zllgamma",
    "Wenu",
    "Wtaunu",
]

FEATURES = [
    "n_jet", "n_bjet", "n_e_medium", "n_mu",
    "ph_pt", "ph_eta", "ph_phi",
    "jet_lead_pt", "jet_lead_eta", "jet_lead_phi", "jet_lead_E",
    "jet_sublead_pt", "jet_sublead_eta", "jet_sublead_phi", "jet_sublead_E",
    "metTST_pt", "metTST_phi", "metTSTsignif",
    "soft_term_pt", "soft_term_phi", "jet_sum_pt",
]

METADATA = [
    "event_uid", "process_group", "campaign", "mc_channel_number",
    "label_binary", "weight_physics", "weight_abs_physics",
]
SPLITS = ["train", "validation", "test"]
MODELS = ["XGBoost", "LightGBM"]

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cpu",
    "eta": 0.08,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "lambda": 5.0,
    "alpha": 0.0,
    "gamma": 0.0,
    "max_bin": 128,
    "seed": SEED,
    "nthread": THREADS,
    "verbosity": 0,
}
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.08,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "lambda_l2": 5.0,
    "lambda_l1": 0.0,
    "max_depth": 8,
    "max_bin": 128,
    "feature_pre_filter": False,
    "deterministic": True,
    "force_col_wise": True,
    "seed": SEED,
    "bagging_seed": SEED,
    "feature_fraction_seed": SEED,
    "data_random_seed": SEED,
    "num_threads": THREADS,
    "verbosity": -1,
}


def save_json(value, name):
    text = json.dumps(value, ensure_ascii=False, indent=2)
    (OUT / name).write_text(text, encoding="utf-8")


def save_table(rows, name):
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "tables" / (name + ".csv"), index=False,
                 encoding="utf-8-sig")
    return table


# %% 2. Загрузка только указанных MC-файлов и готового разбиения по UID.
def load_events():
    manifest = pd.read_csv(DATA / "split_manifest.csv", engine="pyarrow",
                           dtype={"event_uid": "string[pyarrow]"})
    assert manifest.event_uid.notna().all()
    assert manifest.event_uid.is_unique
    assert set(manifest.split) == set(SPLITS)
    manifest = manifest.set_index("event_uid")

    frames = []
    for process in PROCESSES:
        path = DATA / "parquet" / (process + ".parquet")
        frame = pd.read_parquet(path, columns=METADATA + FEATURES)
        assert frame.event_uid.notna().all()
        assert frame.event_uid.is_unique
        assert frame.process_group.eq(process).all()
        assert frame.label_binary.eq(int(process == SIGNAL)).all()
        assert np.isfinite(frame[["weight_physics", "weight_abs_physics"]].to_numpy()).all()
        assert np.allclose(frame.weight_abs_physics, abs(frame.weight_physics),
                           rtol=1e-12, atol=1e-12)

        # reindex сохраняет порядок событий в файле. Случайного split нет.
        assigned = manifest.reindex(frame.event_uid)
        assert assigned.split.notna().all()
        for column in ["process_group", "campaign", "mc_channel_number"]:
            assert np.array_equal(frame[column], assigned[column])
        frame["split"] = assigned.split.to_numpy()
        for column in ["process_group", "campaign", "split"]:
            frame[column] = frame[column].astype("category")
        frames.append(frame)
        print(f"{process}: {len(frame):,} событий", flush=True)

    events = pd.concat(frames, ignore_index=True)
    assert events.event_uid.is_unique
    assert len(events) == len(manifest)
    counts = pd.crosstab(events.process_group, events.split)
    counts = counts.reindex(index=PROCESSES, columns=SPLITS)
    counts["total"] = counts.sum(axis=1)
    for split in SPLITS:
        counts["f_" + split] = counts[split] / counts.total
    save_table(counts.reset_index(), "counts_by_process_split")
    campaigns = pd.crosstab(events.campaign, events.split)
    save_table(campaigns.reset_index(), "counts_by_campaign_split")
    return events


def prepare_samples(events):
    samples = {}
    for split in SPLITS:
        frame = events.loc[events.split == split]
        features = frame[FEATURES].to_numpy(dtype=np.float32, copy=True)
        features[~np.isfinite(features)] = np.nan
        samples[split] = {
            "X": features,
            "y": frame.label_binary.to_numpy(dtype=np.int8),
            "physics": frame.weight_physics.to_numpy(dtype=np.float64),
            "abs_weight": frame.weight_abs_physics.to_numpy(dtype=np.float64),
            "uid": frame.event_uid.reset_index(drop=True),
            "process": frame.process_group.reset_index(drop=True),
        }
        assert set(samples[split]["y"]) == {0, 1}
    return samples


# %% 3. Один коэффициент для сигнала и один для совокупного фона.
def training_weights(samples):
    train = samples["train"]
    weights = train["abs_weight"].copy()
    factors = {}
    signed_factors = {}
    for label in [1, 0]:
        selected = train["y"] == label
        factors[label] = len(weights) / (2 * weights[selected].sum())
        signed_factors[label] = len(weights) / (2 * train["physics"][selected].sum())
        weights[selected] *= factors[label]
        assert np.isclose(weights[selected].sum(), len(weights) / 2, rtol=1e-12)

    rows = []
    for split in SPLITS:
        sample = samples[split]
        for label, name in [(1, "signal"), (0, "background")]:
            selected = sample["y"] == label
            signed = sample["physics"][selected]
            absolute = sample["abs_weight"][selected]
            row = {
                "split": split, "class": name, "n": len(signed),
                "n_negative": int((signed < 0).sum()),
                "sum_physics": signed.sum(), "sum_abs_physics": absolute.sum(),
                "sum_w2": np.square(signed).sum(),
                "signed_to_abs": signed.sum() / absolute.sum(),
                "sum_training_weight": np.nan,
            }
            if split == "train":
                row["sum_training_weight"] = weights[selected].sum()
            rows.append(row)
    save_table(rows, "weights_by_class_split")
    save_json({"k_abs": factors, "k_signed": signed_factors},
              "weight_normalization.json")
    return weights, factors, signed_factors


# %% 4. Гистограммы. H = сумма весов, Q = сумма квадратов весов.
def histogram(values, weights, edges):
    finite = np.isfinite(values)
    folded = np.clip(values[finite], edges[0], np.nextafter(edges[-1], edges[0]))
    weights_finite = weights[finite]
    counts = np.histogram(folded, bins=edges)[0]
    sum_w = np.histogram(folded, bins=edges, weights=weights_finite)[0]
    sum_w2 = np.histogram(folded, bins=edges, weights=weights_finite**2)[0]
    integral = sum_w.sum()
    assert np.isfinite(integral) and integral > 0
    return pd.DataFrame({
        "bin": np.arange(len(counts)), "left": edges[:-1], "right": edges[1:],
        "n": counts, "sum_w": sum_w, "sum_w2": sum_w2,
        "integral": integral, "normalized": sum_w / integral,
        "error": np.sqrt(sum_w2) / abs(integral),
    })


def input_histograms(events, samples, factors, signed_factors):
    train = samples["train"]
    train_mask = events.split == "train"
    tables = []
    tails = []
    quality = []
    edges_by_feature = {}
    units = {}
    for feature in FEATURES:
        all_values = events[feature].to_numpy(dtype=np.float64)
        values = all_values[train_mask]
        finite = values[np.isfinite(values)]
        quality.append({
            "feature": feature, "missing_all": int(np.isnan(all_values).sum()),
            "infinite_all": int(np.isinf(all_values).sum()),
            "unique_train": pd.Series(finite).nunique(),
            "min_train": finite.min(), "max_train": finite.max(),
        })
        if feature.startswith("n_"):
            maximum = max(1, int(finite.max()))
            edges = np.arange(-0.5, maximum + 1.5, 1)
            unit = "число"
        elif feature.endswith("_phi"):
            edges = np.linspace(-np.pi, np.pi, 41)
            unit = "rad"
        elif feature.endswith("_eta"):
            edges = np.linspace(-5, 5, 41)
            unit = "безразмерно"
        else:
            minimum = min(0.0, float(finite.min()))
            maximum = max(minimum + 1, float(np.quantile(finite, 0.995)))
            edges = np.linspace(minimum, maximum, 41)
            unit = "GeV (принято)"
            if feature == "metTSTsignif":
                unit = "безразмерно"
        edges_by_feature[feature] = edges.tolist()
        units[feature] = unit

        for label in [1, 0]:
            selected = train["y"] == label
            for variant in ["parquet_abs", "parquet_signed"]:
                if variant == "parquet_abs":
                    weights = train["abs_weight"] * factors[label]
                else:
                    weights = train["physics"] * signed_factors[label]
                class_values = values[selected]
                class_weights = weights[selected]
                table = histogram(class_values, class_weights, edges)
                table["variant"] = variant
                table["feature"] = feature
                table["class"] = label
                tables.append(table)
                row = {"variant": variant, "feature": feature, "class": label}
                masks = {
                    "nonfinite": ~np.isfinite(class_values),
                    "underflow": np.isfinite(class_values) & (class_values < edges[0]),
                    "overflow": np.isfinite(class_values) & (class_values > edges[-1]),
                }
                for tail, mask in masks.items():
                    row[tail + "_n"] = int(mask.sum())
                    row[tail + "_sum_w"] = class_weights[mask].sum()
                    row[tail + "_sum_w2"] = np.square(class_weights[mask]).sum()
                tails.append(row)

    result = pd.concat(tables, ignore_index=True)
    save_table(result, "input_histogram_bins")
    save_table(tails, "input_histogram_tails_missing")
    save_table(quality, "feature_quality")
    save_json({"edges": edges_by_feature, "units": units,
               "range_policy": "train only; fold tails; exclude nonfinite per feature"},
              "histogram_specification.json")
    plots.input_distributions(result, FEATURES, units, OUT / "figures")


# %% 5. Четыре заранее заданных варианта каждого алгоритма.
def train_models(samples, weights):
    train = samples["train"]
    validation = samples["validation"]
    best = {}
    search = []
    histories = {}
    save_json({"xgboost_base": XGB_PARAMS, "lightgbm_base": LGB_PARAMS,
               "max_rounds": MAX_ROUNDS, "patience": PATIENCE,
               "xgboost_grid": {"max_depth": [3, 5], "min_child_weight": [5, 20]},
               "lightgbm_grid": {"num_leaves": [15, 31], "min_data_in_leaf": [100, 300]},
               "train_subset_used": False}, "search_protocol.json")

    xgb_train = xgb.QuantileDMatrix(
        train["X"], label=train["y"], weight=weights,
        feature_names=FEATURES, max_bin=128, nthread=THREADS,
    )
    xgb_validation = xgb.QuantileDMatrix(
        validation["X"], label=validation["y"], weight=validation["abs_weight"],
        feature_names=FEATURES, max_bin=128, nthread=THREADS, ref=xgb_train,
    )
    trial = 0
    for depth in [3, 5]:
        for child_weight in [5, 20]:
            trial += 1
            params = XGB_PARAMS.copy()
            params["max_depth"] = depth
            params["min_child_weight"] = child_weight
            history = {}
            stop = xgb.callback.EarlyStopping(
                rounds=PATIENCE, metric_name="auc", data_name="validation",
                maximize=True, save_best=True,
            )
            started = time.perf_counter()
            model = xgb.train(
                params, xgb_train, num_boost_round=MAX_ROUNDS,
                evals=[(xgb_validation, "validation")], evals_result=history,
                callbacks=[stop], verbose_eval=False,
            )
            seconds = time.perf_counter() - started
            score = model.predict(xgb_validation)
            auc = roc_auc_score(validation["y"], score,
                                sample_weight=validation["abs_weight"])
            histories["XGBoost_" + str(trial)] = history
            search.append({"model": "XGBoost", "trial": trial,
                           "validation_auc": auc, "rounds": model.num_boosted_rounds(),
                           "round_cap_reached": len(history["validation"]["auc"]) == MAX_ROUNDS,
                           "fit_seconds": seconds, "params": json.dumps(params, sort_keys=True)})
            if "XGBoost" not in best or auc > best["XGBoost"]["validation_auc"]:
                best["XGBoost"] = {"model": model, "params": params,
                                   "validation_auc": auc, "score": score,
                                   "rounds": model.num_boosted_rounds()}
            save_table(search, "hyperparameter_search")
            print(f"XGBoost {trial}/4: validation AUC={auc:.9f}", flush=True)
    del xgb_train, xgb_validation
    gc.collect()

    dataset_params = {"max_bin": 128, "feature_pre_filter": False, "data_random_seed": SEED}
    lgb_train = lgb.Dataset(train["X"], label=train["y"], weight=weights,
                            feature_name=FEATURES, params=dataset_params)
    lgb_validation = lgb.Dataset(
        validation["X"], label=validation["y"], weight=validation["abs_weight"],
        reference=lgb_train, feature_name=FEATURES,
    )
    trial = 0
    for leaves in [15, 31]:
        for min_leaf in [100, 300]:
            trial += 1
            params = LGB_PARAMS.copy()
            params["num_leaves"] = leaves
            params["min_data_in_leaf"] = min_leaf
            history = {}
            started = time.perf_counter()
            model = lgb.train(
                params, lgb_train, num_boost_round=MAX_ROUNDS,
                valid_sets=[lgb_validation], valid_names=["validation"],
                callbacks=[lgb.early_stopping(PATIENCE, first_metric_only=True, verbose=False),
                           lgb.record_evaluation(history)],
            )
            seconds = time.perf_counter() - started
            score = model.predict(validation["X"], num_iteration=model.best_iteration,
                                   num_threads=THREADS)
            auc = roc_auc_score(validation["y"], score,
                                sample_weight=validation["abs_weight"])
            rounds = model.best_iteration or model.current_iteration()
            histories["LightGBM_" + str(trial)] = history
            search.append({"model": "LightGBM", "trial": trial,
                           "validation_auc": auc, "rounds": rounds,
                           "round_cap_reached": len(history["validation"]["auc"]) == MAX_ROUNDS,
                           "fit_seconds": seconds, "params": json.dumps(params, sort_keys=True)})
            if "LightGBM" not in best or auc > best["LightGBM"]["validation_auc"]:
                best["LightGBM"] = {"model": model, "params": params,
                                    "validation_auc": auc, "score": score, "rounds": rounds}
            save_table(search, "hyperparameter_search")
            print(f"LightGBM {trial}/4: validation AUC={auc:.9f}", flush=True)

    save_json(histories, "learning_histories.json")
    plots.learning_curves(histories, OUT / "figures")
    xgb_path = OUT / "models" / "xgboost.ubj"
    xgb_path.write_bytes(best["XGBoost"]["model"].save_raw(raw_format="ubj"))
    lgb_text = best["LightGBM"]["model"].model_to_string(
        num_iteration=best["LightGBM"]["rounds"])
    (OUT / "models" / "lightgbm.txt").write_text(lgb_text, encoding="utf-8")
    return best


# %% 6. Порог: максимум ZA только на validation, с физическими знаковыми весами.
def significance(signal, background):
    signal = np.asarray(signal, dtype=float)
    background = np.asarray(background, dtype=float)
    result = np.full(signal.shape, np.nan)
    valid = np.isfinite(signal) & np.isfinite(background) & (signal >= 0) & (background > 0)
    s = signal[valid]
    b = background[valid]
    result[valid] = np.sqrt(np.maximum(0, 2 * ((s + b) * np.log1p(s / b) - s)))
    return result


def threshold_scan(score, sample):
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    labels = sample["y"][order]
    weights = sample["physics"][order] / HOLDOUT_FRACTION
    # Все события с одинаковым score включаем вместе.
    ends = np.flatnonzero(sorted_score[:-1] != sorted_score[1:])
    ends = np.append(ends, len(score) - 1)
    table = pd.DataFrame({"threshold": sorted_score[ends]})
    for label, name, yield_name in [(1, "signal", "S"), (0, "background", "B")]:
        selected = labels == label
        table[yield_name] = np.cumsum(weights * selected, dtype=np.float64)[ends]
        table["sum_w2_" + name] = np.cumsum(weights**2 * selected, dtype=np.float64)[ends]
        table["abs_sum_" + name] = np.cumsum(abs(weights) * selected, dtype=np.float64)[ends]
        table["n_" + name] = np.cumsum(selected, dtype=np.int64)[ends]
    table["ZA"] = significance(table.S, table.B)
    return table


def physical_yields(score, sample, threshold):
    result = {"threshold": float(threshold)}
    passed = score >= threshold
    for label, name, yield_name in [(1, "signal", "S"), (0, "background", "B")]:
        selected = passed & (sample["y"] == label)
        weights = sample["physics"][selected] / HOLDOUT_FRACTION
        total = float(weights.sum())
        squares = float(np.square(weights).sum())
        absolute = float(abs(weights).sum())
        result[yield_name] = total
        result["n_" + name] = len(weights)
        result["sum_w2_" + name] = squares
        result["abs_sum_" + name] = absolute
        result["neff_" + name] = 0.0
        if squares > 0:
            result["neff_" + name] = total**2 / squares
        result["signed_to_abs_" + name] = np.nan
        if absolute > 0:
            result["signed_to_abs_" + name] = abs(total) / absolute
        result["mc_relative_error_" + name] = np.inf
        if total != 0:
            result["mc_relative_error_" + name] = float(np.sqrt(squares) / abs(total))
    result["ZA"] = float(significance(result["S"], result["B"]))
    return result


def select_thresholds(best, samples):
    validation = samples["validation"]
    scans = {}
    stability = []
    for name in MODELS:
        score = best[name]["score"]
        assert np.isfinite(score).all()
        scan = threshold_scan(score, validation)
        assert scan.ZA.notna().any()
        index = scan.ZA.idxmax()
        threshold = float(scan.loc[index, "threshold"])
        yields = physical_yields(score, validation, threshold)
        assert np.allclose([yields["S"], yields["B"]], scan.loc[index, ["S", "B"]],
                           rtol=1e-9, atol=1e-8)
        yields["unstable_mc_flag"] = (
            yields["neff_background"] < 25 or yields["neff_signal"] < 25
            or yields["signed_to_abs_background"] < 0.5
        )
        best[name]["threshold"] = threshold
        best[name]["validation_yields"] = yields
        scan.to_parquet(OUT / "tables" / (name.lower() + "_validation_threshold_scan.parquet"),
                        index=False)
        scans[name] = scan

        # Диагностика устойчивости: удаляем по одному 20 самых весомых событий.
        kept = np.flatnonzero(score >= threshold)
        influential = np.argsort(-abs(validation["physics"][kept]))[:20]
        for event in kept[influential]:
            delta = float(validation["physics"][event] / HOLDOUT_FRACTION)
            signal = yields["S"]
            background = yields["B"]
            if validation["y"][event] == 1:
                signal -= delta
            else:
                background -= delta
            removed = float(significance(signal, background))
            stability.append({
                "model": name, "event_uid": str(validation["uid"].iloc[event]),
                "class": int(validation["y"][event]), "scaled_signed_weight": delta,
                "ZA_original": yields["ZA"], "ZA_without_event": removed,
                "delta_ZA": removed - yields["ZA"],
            })
    save_table(stability, "threshold_leave_one_event_out")
    plots.thresholds(scans, best, OUT / "figures")

    champion = "XGBoost"
    if best["LightGBM"]["validation_auc"] > best["XGBoost"]["validation_auc"]:
        champion = "LightGBM"
    frozen = {
        "selected_model_by_validation_auc": champion, "test_used_for_selection": False,
        "feature_names": FEATURES, "seed": SEED, "models": {},
    }
    for split in SPLITS:
        frozen[split + "_events"] = len(samples[split]["y"])
    for name in MODELS:
        frozen["models"][name] = {
            "params": best[name]["params"], "rounds": best[name]["rounds"],
            "validation_auc": best[name]["validation_auc"],
            "threshold": best[name]["threshold"],
            "validation_yields": best[name]["validation_yields"],
        }
    # Этот файл записывается до первого вычисления test score.
    save_json(frozen, "frozen_selection_before_test.json")
    print(f"Выбор по validation: {champion}. Модели и пороги зафиксированы.", flush=True)
    return champion


# %% 7. Итоговая оценка. Решения выше уже приняты без test.
def evaluate(best, samples):
    scores = {"train": {}, "validation": {}, "test": {}}
    metrics = []
    yields = []
    curves = []
    process_yields = []
    for name in MODELS:
        model = best[name]["model"]
        scores["validation"][name] = best[name]["score"]
        for split in ["train", "test"]:
            features = samples[split]["X"]
            if name == "XGBoost":
                score = model.inplace_predict(features)
            else:
                score = model.predict(features, num_iteration=best[name]["rounds"],
                                       num_threads=THREADS)
            assert np.isfinite(score).all()
            scores[split][name] = score

        for split in SPLITS:
            sample = samples[split]
            score = scores[split][name]
            auc = roc_auc_score(sample["y"], score, sample_weight=sample["abs_weight"])
            metrics.append({"model": name, "split": split, "weighted_auc": auc,
                            "n_events": len(score)})
            fpr, tpr, thresholds = roc_curve(sample["y"], score,
                                             sample_weight=sample["abs_weight"],
                                             drop_intermediate=False)
            curves.append(pd.DataFrame({"model": name, "split": split,
                                        "FPR": fpr, "TPR": tpr, "threshold": thresholds}))
            if split == "train":
                continue
            row = physical_yields(score, sample, best[name]["threshold"])
            row["model"] = name
            row["split"] = split
            yields.append(row)
            passed = score >= best[name]["threshold"]
            for process in PROCESSES:
                selected = passed & sample["process"].eq(process).to_numpy()
                weights = sample["physics"][selected] / HOLDOUT_FRACTION
                squares = float(np.square(weights).sum())
                process_yields.append({
                    "model": name, "split": split, "process_group": process,
                    "n_selected": int(selected.sum()), "physics_yield": weights.sum(),
                    "sum_w2_scaled": squares, "mc_error": np.sqrt(squares),
                })

    metrics = save_table(metrics, "final_metrics")
    yields = save_table(yields, "physical_yields_at_frozen_threshold")
    save_table(process_yields, "physical_yields_by_process")
    roc = pd.concat(curves, ignore_index=True)
    roc.to_parquet(OUT / "tables" / "roc_curves.parquet", index=False)
    plots.roc_curves(roc, metrics, OUT / "figures")
    for split in ["validation", "test"]:
        sample = samples[split]
        predictions = pd.DataFrame({
            "event_uid": sample["uid"], "process_group": sample["process"],
            "split": split, "label_binary": sample["y"],
            "weight_physics": sample["physics"], "weight_abs_physics": sample["abs_weight"],
            "score_xgboost": scores[split]["XGBoost"],
            "score_lightgbm": scores[split]["LightGBM"],
        })
        predictions.to_parquet(OUT / "predictions" / (split + "_predictions.parquet"),
                               index=False)
    return scores, metrics, yields


# %% 8. Отклик Train/Test и полная ошибка отношения двух гистограмм.
def response_histograms(scores, samples):
    edges = np.linspace(0, 1, 41)
    tables = []
    ratios = []
    for name in MODELS:
        for label in [1, 0]:
            histograms = {}
            for split in ["train", "test"]:
                sample = samples[split]
                selected = sample["y"] == label
                table = histogram(scores[split][name][selected],
                                  sample["abs_weight"][selected], edges)
                assert np.isclose(table.normalized.sum(), 1, rtol=1e-12)
                table["model"] = name
                table["class"] = label
                table["split"] = split
                histograms[split] = table
                tables.append(table)

            train = histograms["train"]
            test = histograms["test"]
            ratio = np.full(len(train), np.nan)
            error = np.full(len(train), np.nan)
            valid = train.normalized != 0
            denominator = train.normalized[valid].to_numpy()
            numerator = test.normalized[valid].to_numpy()
            train_error = train.error[valid].to_numpy()
            test_error = test.error[valid].to_numpy()
            ratio[valid] = numerator / denominator
            error[valid] = np.sqrt(test_error**2 / denominator**2
                                   + numerator**2 * train_error**2 / denominator**4)
            ratios.append(pd.DataFrame({
                "model": name, "class": label, "bin": train.bin,
                "left": train.left, "right": train.right,
                "ratio": ratio, "full_error": error, "zero_train_denominator": ~valid,
            }))
    tables = save_table(pd.concat(tables, ignore_index=True), "score_histogram_bins")
    ratios = save_table(pd.concat(ratios, ignore_index=True), "test_train_ratio_bins")
    plots.responses(tables, ratios, OUT / "figures")


# %% 9. Выполнение этапов по порядку.
def main():
    for folder in ["tables", "figures", "models", "predictions"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in ["numpy", "pandas", "pyarrow", "xgboost", "lightgbm", "scikit-learn"]:
        versions[package] = importlib.metadata.version(package)
    save_json({"python": platform.python_version(), "versions": versions,
               "seed": SEED, "threads_per_model": THREADS, "feature_names": FEATURES,
               "device": "CPU", "holdout_fraction_for_yields": HOLDOUT_FRACTION},
              "environment.json")

    started = time.perf_counter()
    events = load_events()
    samples = prepare_samples(events)
    weights, factors, signed_factors = training_weights(samples)
    print("Строю распределения всех 21 входных признаков.", flush=True)
    input_histograms(events, samples, factors, signed_factors)
    del events
    gc.collect()
    print("Обучаю восемь вариантов моделей на полном train.", flush=True)
    best = train_models(samples, weights)
    champion = select_thresholds(best, samples)
    frozen_path = OUT / "frozen_selection_before_test.json"
    frozen_before_test = frozen_path.read_bytes()
    scores, metrics, yields = evaluate(best, samples)
    response_histograms(scores, samples)
    assert frozen_path.read_bytes() == frozen_before_test
    selected = yields.loc[(yields.model == champion) & (yields.split == "test")].iloc[0]
    test_auc = metrics.loc[(metrics.model == champion) & (metrics.split == "test"),
                           "weighted_auc"].iloc[0]
    save_json({"selected_model": champion, "test_auc": float(test_auc),
               "threshold": float(selected.threshold), "test_used_for_selection": False,
               "test_yields": selected.to_dict(), "n_events": int(sum(metrics.n_events) // 2),
               "runtime_seconds": time.perf_counter() - started}, "summary.json")
    print(metrics.to_string(index=False), flush=True)
    print(yields[["model", "split", "threshold", "S", "B", "ZA"]].to_string(index=False),
          flush=True)
    print(f"Готово. Результаты: {OUT}", flush=True)


if __name__ == "__main__":
    main()
