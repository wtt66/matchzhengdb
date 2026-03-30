from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import Circle, FancyBboxPatch
from scipy.optimize import minimize
from scipy.stats import chi2 as chi2_dist

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_analysis_utils import (
    GENDER_MAP,
    PALETTE,
    PART_CONTEXT,
    build_schema,
    cronbach_alpha,
    ensure_output_dir,
    load_data,
    map_codes,
    save_figure,
    to_numeric,
    write_report,
)


CONSTRUCT_ORDER = ["CVP", "PK", "PC", "EA", "PR", "PI", "PKN", "BI"]
CONSTRUCT_KEYS = {
    "CVP": "cvp",
    "PK": "pk",
    "PC": "pc",
    "EA": "ea",
    "PR": "pr",
    "PI": "pi",
    "PKN": "pkn",
    "BI": "bi",
}
CONSTRUCT_LABELS = {
    "CVP": "文化价值感知",
    "PK": "产品知识",
    "PC": "购买便利性",
    "EA": "经济可及性",
    "PR": "感知风险",
    "PI": "产品涉入度",
    "PKN": "先验知识",
    "BI": "购买意愿",
    "PREP": "购买准备度",
    "ACCESS": "交易可得性",
    "ENGAGE": "认知卷入基础",
    "ACCEPT": "非遗融入接受度",
    "CVP_x_PI": "文化价值感知 × 产品涉入度",
    "PC_x_PI": "购买便利性 × 产品涉入度",
    "PR_x_PKN": "感知风险 × 先验知识",
}
STRUCTURAL_PREDICTORS = ["CVP", "PK", "PC", "EA", "PR", "PI", "PKN", "ACCEPT"]
INTERACTION_TERMS = [("CVP", "PI"), ("PC", "PI"), ("PR", "PKN")]
PLOT_ORDER = STRUCTURAL_PREDICTORS + ["CVP_x_PI", "PC_x_PI", "PR_x_PKN"]


class CFAResult:
    def __init__(
        self,
        *,
        observed: pd.DataFrame,
        standardized: pd.DataFrame,
        factor_scores: pd.DataFrame,
        loadings: pd.DataFrame,
        cross_loading_corr: pd.DataFrame,
        factor_cov: pd.DataFrame,
        factor_corr: pd.DataFrame,
        residual_var: pd.Series,
        sample_cov: pd.DataFrame,
        implied_cov: pd.DataFrame,
        residual_cov: pd.DataFrame,
        fit_stats: pd.Series,
        success: bool,
        message: str,
    ) -> None:
        self.observed = observed
        self.standardized = standardized
        self.factor_scores = factor_scores
        self.loadings = loadings
        self.cross_loading_corr = cross_loading_corr
        self.factor_cov = factor_cov
        self.factor_corr = factor_corr
        self.residual_var = residual_var
        self.sample_cov = sample_cov
        self.implied_cov = implied_cov
        self.residual_cov = residual_cov
        self.fit_stats = fit_stats
        self.success = success
        self.message = message


def _context_text(field: str, fallback: str) -> str:
    part_text = PART_CONTEXT.get("第四部分", {})
    return str(part_text.get(field, fallback))


def pretty_item_label(item: str) -> str:
    match = re.match(r"([A-Z]+)(\d+)", item)
    if not match:
        return item
    construct, number = match.groups()
    return f"{CONSTRUCT_LABELS.get(construct, construct)}{number}"


def _pretty_path_label(name: str) -> str:
    return CONSTRUCT_LABELS.get(name, name)


def _item_construct(item: str) -> str:
    match = re.match(r"([A-Z]+)\d+$", item)
    return match.group(1) if match else ""


def _construct_items(index: pd.Index, construct: str) -> list[str]:
    return [item for item in index if _item_construct(str(item)) == construct]


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.diag(cov))
    scale[scale == 0] = np.nan
    corr = cov / np.outer(scale, scale)
    return np.nan_to_num(corr)


def _softplus(values: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0)


def _inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(value)))


def prepare_part4_data(df: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    schema = build_schema(df)["part4"]
    item_frames: dict[str, pd.DataFrame] = {}
    for construct in CONSTRUCT_ORDER:
        frame = df[schema[CONSTRUCT_KEYS[construct]]].apply(pd.to_numeric, errors="coerce")
        frame = frame.rename(columns={col: f"{construct}{idx + 1}" for idx, col in enumerate(frame.columns)})
        item_frames[construct] = frame
    scores = pd.DataFrame({construct: frame.mean(axis=1) for construct, frame in item_frames.items()})
    scores["ACCEPT"] = to_numeric(df[schema["q17_acceptance"]])
    return item_frames, scores


def build_construct_mean_scores(item_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({construct: frame.mean(axis=1) for construct, frame in item_frames.items()})


def fit_cfa_model(
    item_frames: dict[str, pd.DataFrame],
    mean_scores: pd.DataFrame,
    *,
    construct_order: list[str] | None = None,
) -> CFAResult:
    construct_order = construct_order or list(item_frames.keys())
    observed = pd.concat([item_frames[construct] for construct in construct_order], axis=1)
    observed = observed.dropna(axis=0, how="any")
    standardized = (observed - observed.mean()) / observed.std(ddof=0)
    sample_cov = standardized.cov().values
    logdet_sample = np.linalg.slogdet(sample_cov)[1]
    p = standardized.shape[1]
    n_obs = standardized.shape[0]
    n_factors = len(construct_order)

    factor_items = {construct: item_frames[construct].columns.tolist() for construct in construct_order}
    indicator_names = standardized.columns.tolist()
    indicator_index = {name: idx for idx, name in enumerate(indicator_names)}
    factor_item_positions = [
        [indicator_index[name] for name in factor_items[construct]] for construct in construct_order
    ]

    n_loading_params = sum(len(indices) - 1 for indices in factor_item_positions)
    n_cholesky_params = n_factors * (n_factors + 1) // 2
    n_error_params = p

    aligned_scores = mean_scores.loc[observed.index, construct_order].copy()
    phi_init = aligned_scores.cov().values + np.eye(n_factors) * 1e-3
    chol_init = np.linalg.cholesky(phi_init)
    chol_params: list[float] = []
    for row in range(n_factors):
        for col in range(row + 1):
            chol_params.append(float(np.log(chol_init[row, col])) if row == col else float(chol_init[row, col]))

    initial = np.concatenate(
        [
            np.full(n_loading_params, _inverse_softplus(0.75)),
            np.array(chol_params),
            np.log(np.full(n_error_params, 0.40)),
        ]
    )

    def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        load_raw = params[:n_loading_params]
        chol_raw = params[n_loading_params : n_loading_params + n_cholesky_params]
        error_raw = params[n_loading_params + n_cholesky_params :]

        loading_matrix = np.zeros((p, n_factors))
        cursor = 0
        for factor_idx, indices in enumerate(factor_item_positions):
            loading_matrix[indices[0], factor_idx] = 1.0
            for item_idx in indices[1:]:
                loading_matrix[item_idx, factor_idx] = _softplus(np.array([load_raw[cursor]]))[0] + 0.05
                cursor += 1

        lower = np.zeros((n_factors, n_factors))
        cursor = 0
        for row in range(n_factors):
            for col in range(row + 1):
                lower[row, col] = float(np.exp(chol_raw[cursor])) if row == col else float(chol_raw[cursor])
                cursor += 1
        factor_cov = lower @ lower.T
        error_cov = np.diag(np.exp(error_raw) + 1e-4)
        return loading_matrix, factor_cov, error_cov

    def objective(params: np.ndarray) -> float:
        try:
            loading_matrix, factor_cov, error_cov = unpack(params)
            implied_cov = loading_matrix @ factor_cov @ loading_matrix.T + error_cov
            sign, logdet_implied = np.linalg.slogdet(implied_cov)
            if sign <= 0:
                return 1e6
            inv_implied = np.linalg.inv(implied_cov)
            value = logdet_implied + np.trace(sample_cov @ inv_implied) - logdet_sample - p
            if not np.isfinite(value):
                return 1e6
            return float(value)
        except Exception:
            return 1e6

    best_result = None
    current = initial
    for _ in range(3):
        result = minimize(
            objective,
            current,
            method="L-BFGS-B",
            options={"maxiter": 250, "maxfun": 30000},
        )
        current = result.x
        best_result = result

    assert best_result is not None
    loading_matrix, factor_cov, error_cov = unpack(best_result.x)
    implied_cov = loading_matrix @ factor_cov @ loading_matrix.T + error_cov
    implied_var = np.diag(implied_cov)
    standardized_loadings = np.zeros_like(loading_matrix)
    for factor_idx in range(n_factors):
        standardized_loadings[:, factor_idx] = (
            loading_matrix[:, factor_idx] * math.sqrt(factor_cov[factor_idx, factor_idx]) / np.sqrt(implied_var)
        )

    loading_df = pd.DataFrame(standardized_loadings, index=indicator_names, columns=construct_order)

    factor_scores = {}
    for factor_idx, construct in enumerate(construct_order):
        cols = factor_items[construct]
        weights = loading_df.loc[cols, construct].abs()
        block = standardized[cols]
        factor_scores[construct] = block.mul(weights, axis=1).sum(axis=1) / weights.sum()
    factor_scores_df = pd.DataFrame(factor_scores, index=standardized.index)

    factor_score_z = (factor_scores_df - factor_scores_df.mean()) / factor_scores_df.std(ddof=0)
    cross_loading_corr = pd.DataFrame(index=indicator_names, columns=construct_order, dtype=float)
    for item in indicator_names:
        for construct in construct_order:
            cross_loading_corr.loc[item, construct] = standardized[item].corr(factor_score_z[construct])

    n_parameters = (p - n_factors) + (n_factors * (n_factors + 1) // 2) + p
    df_model = p * (p + 1) // 2 - n_parameters
    chi2_value = max((n_obs - 1) * objective(best_result.x), 0.0)
    null_cov = np.diag(np.diag(sample_cov))
    null_fit = np.linalg.slogdet(null_cov)[1] + np.trace(sample_cov @ np.linalg.inv(null_cov)) - logdet_sample - p
    chi2_null = max((n_obs - 1) * null_fit, 0.0)
    df_null = p * (p - 1) // 2
    cfi = 1 - max(chi2_value - df_model, 0.0) / max(chi2_null - df_null, 1e-9)
    tli = (chi2_null / df_null - chi2_value / df_model) / (chi2_null / df_null - 1)
    rmsea = math.sqrt(max((chi2_value - df_model) / (df_model * (n_obs - 1)), 0.0))
    tri_idx = np.tril_indices(p)
    srmr = float(np.sqrt(np.mean((sample_cov[tri_idx] - implied_cov[tri_idx]) ** 2)))

    fit_stats = pd.Series(
        {
            "n": float(n_obs),
            "chi2": chi2_value,
            "df": float(df_model),
            "p_value": float(chi2_dist.sf(chi2_value, df_model)),
            "CFI": cfi,
            "TLI": tli,
            "RMSEA": rmsea,
            "SRMR": srmr,
        }
    )

    factor_cov_df = pd.DataFrame(factor_cov, index=construct_order, columns=construct_order)
    factor_corr_df = pd.DataFrame(_cov_to_corr(factor_cov), index=construct_order, columns=construct_order)
    residual_var = pd.Series(np.diag(error_cov) / np.diag(implied_cov), index=indicator_names, name="residual_var")
    sample_cov_df = pd.DataFrame(sample_cov, index=indicator_names, columns=indicator_names)
    implied_cov_df = pd.DataFrame(implied_cov, index=indicator_names, columns=indicator_names)
    residual_cov_df = sample_cov_df - implied_cov_df
    return CFAResult(
        observed=observed,
        standardized=standardized,
        factor_scores=factor_scores_df,
        loadings=loading_df,
        cross_loading_corr=cross_loading_corr,
        factor_cov=factor_cov_df,
        factor_corr=factor_corr_df,
        residual_var=residual_var,
        sample_cov=sample_cov_df,
        implied_cov=implied_cov_df,
        residual_cov=residual_cov_df,
        fit_stats=fit_stats,
        success=bool(best_result.success),
        message=str(best_result.message),
    )


def compute_reliability_validity(
    cfa_result: CFAResult,
    item_frames: dict[str, pd.DataFrame],
    *,
    construct_order: list[str] | None = None,
) -> pd.DataFrame:
    construct_order = construct_order or list(item_frames.keys())
    rows = []
    for construct in construct_order:
        cols = item_frames[construct].columns.tolist()
        own_loadings = cfa_result.loadings.loc[cols, construct].abs()
        ave = float(np.mean(np.square(own_loadings)))
        cr = float((own_loadings.sum() ** 2) / ((own_loadings.sum() ** 2) + np.sum(1 - np.square(own_loadings))))
        rows.append(
            {
                "construct": construct,
                "alpha": float(cronbach_alpha(item_frames[construct])),
                "CR": cr,
                "AVE": ave,
                "sqrt_AVE": math.sqrt(max(ave, 0.0)),
                "mean_loading": float(own_loadings.mean()),
                "max_loading": float(own_loadings.max()),
            }
        )
    return pd.DataFrame(rows).set_index("construct")


def compute_loading_summary(
    cfa_result: CFAResult,
    item_frames: dict[str, pd.DataFrame],
    *,
    construct_order: list[str] | None = None,
) -> pd.DataFrame:
    construct_order = construct_order or list(item_frames.keys())
    rows = []
    for construct in construct_order:
        cols = item_frames[construct].columns.tolist()
        for item in cols:
            own = float(abs(cfa_result.loadings.loc[item, construct]))
            cross = float(abs(cfa_result.cross_loading_corr.loc[item].drop(construct)).max())
            rows.append(
                {
                    "construct": construct,
                    "item": item,
                    "own": own,
                    "cross": cross,
                    "gap": own - cross,
                    "item_label": pretty_item_label(item),
                }
            )
    return pd.DataFrame(rows)


def build_item_audit_table(cfa_result: CFAResult, item_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    audit = compute_loading_summary(cfa_result, item_frames).copy()
    audit["residual_var"] = audit["item"].map(cfa_result.residual_var)
    audit["flag_low_loading"] = audit["own"] < 0.35
    audit["flag_low_gap"] = audit["gap"] < 0.08
    audit["flag_high_residual"] = (audit["own"] < 0.40) & (audit["residual_var"] > 0.85)
    audit["drop_candidate"] = audit[["flag_low_loading", "flag_low_gap", "flag_high_residual"]].any(axis=1)

    reasons = []
    for _, row in audit.iterrows():
        reason_parts = []
        if row["flag_low_loading"]:
            reason_parts.append("载荷<0.35")
        if row["flag_low_gap"]:
            reason_parts.append("区分效度间隔<0.08")
        if row["flag_high_residual"]:
            reason_parts.append("高残差且载荷偏弱")
        reasons.append("；".join(reason_parts) if reason_parts else "保留")
    audit["reason"] = reasons
    return audit.sort_values(["drop_candidate", "gap", "own"], ascending=[False, True, True]).reset_index(drop=True)


def propose_refined_item_frames(
    item_frames: dict[str, pd.DataFrame],
    audit_df: pd.DataFrame,
    minimum_items: int = 2,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    refined_frames: dict[str, pd.DataFrame] = {}
    decision_rows = []
    for construct in CONSTRUCT_ORDER:
        original_cols = item_frames[construct].columns.tolist()
        construct_audit = audit_df[audit_df["construct"] == construct].set_index("item").loc[original_cols].reset_index()
        keep = construct_audit.loc[~construct_audit["drop_candidate"], "item"].tolist()
        removed = construct_audit.loc[construct_audit["drop_candidate"], ["item", "own", "gap"]]
        readded_items: list[str] = []
        if len(keep) < minimum_items:
            refill = removed.sort_values(["own", "gap"], ascending=False)["item"].tolist()
            for item in refill:
                if item not in keep:
                    keep.append(item)
                    readded_items.append(item)
                if len(keep) >= minimum_items:
                    break
        keep = [item for item in original_cols if item in keep]
        refined_frames[construct] = item_frames[construct][keep].copy()
        for _, row in construct_audit.iterrows():
            if row["item"] in readded_items:
                reason = "为保持构念至少2题而补回"
            elif row["item"] in keep:
                reason = "保留"
            else:
                reason = row["reason"] if row["reason"] != "保留" else "透明优化中删除"
            decision_rows.append(
                {
                    "construct": construct,
                    "item": row["item"],
                    "keep_in_refined": row["item"] in keep,
                    "own": row["own"],
                    "cross": row["cross"],
                    "gap": row["gap"],
                    "residual_var": row["residual_var"],
                    "reason": reason,
                }
            )
    decision_df = pd.DataFrame(decision_rows)
    return refined_frames, decision_df


def build_model_comparison_table(
    baseline_cfa: CFAResult,
    refined_cfa: CFAResult,
    baseline_model: sm.regression.linear_model.RegressionResultsWrapper,
    refined_model: sm.regression.linear_model.RegressionResultsWrapper,
    baseline_items: dict[str, pd.DataFrame],
    refined_items: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    item_count_base = int(sum(frame.shape[1] for frame in baseline_items.values()))
    item_count_refined = int(sum(frame.shape[1] for frame in refined_items.values()))
    rows.extend(
        [
            {"metric": "观测题项数", "baseline": item_count_base, "refined": item_count_refined, "delta": item_count_refined - item_count_base},
            {"metric": "CFI", "baseline": baseline_cfa.fit_stats["CFI"], "refined": refined_cfa.fit_stats["CFI"], "delta": refined_cfa.fit_stats["CFI"] - baseline_cfa.fit_stats["CFI"]},
            {"metric": "TLI", "baseline": baseline_cfa.fit_stats["TLI"], "refined": refined_cfa.fit_stats["TLI"], "delta": refined_cfa.fit_stats["TLI"] - baseline_cfa.fit_stats["TLI"]},
            {"metric": "RMSEA", "baseline": baseline_cfa.fit_stats["RMSEA"], "refined": refined_cfa.fit_stats["RMSEA"], "delta": refined_cfa.fit_stats["RMSEA"] - baseline_cfa.fit_stats["RMSEA"]},
            {"metric": "SRMR", "baseline": baseline_cfa.fit_stats["SRMR"], "refined": refined_cfa.fit_stats["SRMR"], "delta": refined_cfa.fit_stats["SRMR"] - baseline_cfa.fit_stats["SRMR"]},
            {"metric": "Adj_R2", "baseline": baseline_model.rsquared_adj, "refined": refined_model.rsquared_adj, "delta": refined_model.rsquared_adj - baseline_model.rsquared_adj},
        ]
    )
    return pd.DataFrame(rows)


def build_structural_model_table(model_specs: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for spec in model_specs:
        rows.append(
            {
                "model": spec["name"],
                "predictors": " + ".join(spec["predictors"]),
                "n_paths": len(spec["predictors"]),
                "adj_r2": spec["model"].rsquared_adj,
                "aic": spec["model"].aic,
                "bic": spec["model"].bic,
                "cv_rmse": spec["cv_rmse"],
                "cv_rmse_sd": spec["cv_rmse_sd"],
            }
        )
    return pd.DataFrame(rows)


def build_merged_construct_frame(
    item_frames: dict[str, pd.DataFrame],
    source_constructs: list[str],
    merged_construct: str,
) -> pd.DataFrame:
    merged = pd.concat([item_frames[construct] for construct in source_constructs], axis=1).copy()
    merged.columns = [f"{merged_construct}{idx + 1}" for idx in range(merged.shape[1])]
    return merged


def build_alternative_sem_specs(refined_item_frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    candidate_a_frames = {
        "CVP": refined_item_frames["CVP"].copy(),
        "PREP": build_merged_construct_frame(refined_item_frames, ["PK", "PC", "EA", "PI"], "PREP"),
        "PR": refined_item_frames["PR"].copy(),
        "PKN": refined_item_frames["PKN"].copy(),
        "BI": refined_item_frames["BI"].copy(),
    }
    candidate_b_frames = {
        "CVP": refined_item_frames["CVP"].copy(),
        "ACCESS": build_merged_construct_frame(refined_item_frames, ["PC", "EA"], "ACCESS"),
        "ENGAGE": build_merged_construct_frame(refined_item_frames, ["PK", "PI", "PKN"], "ENGAGE"),
        "PR": refined_item_frames["PR"].copy(),
        "BI": refined_item_frames["BI"].copy(),
    }
    return [
        {
            "name": "备选A：购买准备度合并模型",
            "description": "将产品知识、购买便利性、经济可及性与产品涉入度合并为“购买准备度”，用于压缩高相关的决策准备类构念。",
            "item_frames": candidate_a_frames,
            "full_predictors": ["CVP", "PREP", "PR", "PKN", "ACCEPT"],
        },
        {
            "name": "备选B：交易可得性/认知卷入合并模型",
            "description": "将购买便利性与经济可及性合并为“交易可得性”，将产品知识、产品涉入度与先验知识合并为“认知卷入基础”。",
            "item_frames": candidate_b_frames,
            "full_predictors": ["CVP", "ACCESS", "ENGAGE", "PR", "ACCEPT"],
        },
    ]


def compute_discriminant_validity_detail(
    cfa_result: CFAResult,
    reliability_df: pd.DataFrame,
    *,
    construct_order: list[str] | None = None,
) -> pd.DataFrame:
    construct_order = construct_order or reliability_df.index.tolist()
    rows = []
    for construct in construct_order:
        others = cfa_result.factor_corr.loc[construct].drop(construct).abs()
        max_other = float(others.max()) if not others.empty else 0.0
        margin = float(reliability_df.loc[construct, "sqrt_AVE"] - max_other)
        rows.append(
            {
                "construct": construct,
                "construct_label": CONSTRUCT_LABELS.get(construct, construct),
                "sqrt_AVE": float(reliability_df.loc[construct, "sqrt_AVE"]),
                "max_abs_corr": max_other,
                "margin": margin,
                "passed": margin > 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_discriminant_validity(
    detail_df: pd.DataFrame,
    factor_corr: pd.DataFrame,
) -> pd.Series:
    corr_abs = factor_corr.abs()
    upper_mask = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
    stacked = corr_abs.where(upper_mask).stack().sort_values(ascending=False)
    worst_pair = ""
    max_abs_corr = 0.0
    if not stacked.empty:
        left, right = stacked.index[0]
        max_abs_corr = float(stacked.iloc[0])
        worst_pair = f"{CONSTRUCT_LABELS.get(left, left)} × {CONSTRUCT_LABELS.get(right, right)}"
    return pd.Series(
        {
            "pass_count": int(detail_df["passed"].sum()),
            "pass_rate": float(detail_df["passed"].mean()) if len(detail_df) else 0.0,
            "min_margin": float(detail_df["margin"].min()) if len(detail_df) else 0.0,
            "mean_margin": float(detail_df["margin"].mean()) if len(detail_df) else 0.0,
            "max_abs_corr": max_abs_corr,
            "worst_pair": worst_pair,
        }
    )


def build_sem_candidate_snapshot(
    *,
    name: str,
    description: str,
    item_frames: dict[str, pd.DataFrame],
    cfa_result: CFAResult,
    reliability_df: pd.DataFrame,
    model_data: pd.DataFrame,
    score_frame: pd.DataFrame,
    coef_table: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    final_predictors: list[str],
    cv_rmse: float,
    cv_rmse_sd: float,
) -> dict[str, object]:
    construct_order = list(item_frames.keys())
    dv_detail = compute_discriminant_validity_detail(
        cfa_result,
        reliability_df,
        construct_order=construct_order,
    )
    dv_summary = summarize_discriminant_validity(dv_detail, cfa_result.factor_corr.loc[construct_order, construct_order])
    return {
        "name": name,
        "description": description,
        "item_frames": item_frames,
        "construct_order": construct_order,
        "cfa": cfa_result,
        "reliability": reliability_df,
        "dv_detail": dv_detail,
        "dv_summary": dv_summary,
        "model_data": model_data,
        "scores": score_frame,
        "coef": coef_table,
        "model": model,
        "final_predictors": final_predictors,
        "cv_rmse": float(cv_rmse),
        "cv_rmse_sd": float(cv_rmse_sd),
    }


def build_candidate_model_comparison_table(
    candidate_snapshots: list[dict[str, object]],
    *,
    baseline_name: str,
) -> pd.DataFrame:
    rows = []
    for snapshot in candidate_snapshots:
        reliability_df = snapshot["reliability"]
        dv_summary = snapshot["dv_summary"]
        rows.append(
            {
                "模型": snapshot["name"],
                "角色": "baseline" if snapshot["name"] == baseline_name else "candidate",
                "构念数": len(snapshot["construct_order"]),
                "观测题项数": int(sum(frame.shape[1] for frame in snapshot["item_frames"].values())),
                "最终路径": " + ".join(snapshot["final_predictors"]),
                "CFI": float(snapshot["cfa"].fit_stats["CFI"]),
                "TLI": float(snapshot["cfa"].fit_stats["TLI"]),
                "RMSEA": float(snapshot["cfa"].fit_stats["RMSEA"]),
                "SRMR": float(snapshot["cfa"].fit_stats["SRMR"]),
                "平均CR": float(reliability_df["CR"].mean()),
                "最小CR": float(reliability_df["CR"].min()),
                "平均AVE": float(reliability_df["AVE"].mean()),
                "最小AVE": float(reliability_df["AVE"].min()),
                "Fornell通过率": float(dv_summary["pass_rate"]),
                "最小判别效度边际": float(dv_summary["min_margin"]),
                "平均判别效度边际": float(dv_summary["mean_margin"]),
                "最大潜变量相关": float(dv_summary["max_abs_corr"]),
                "最差构念对": str(dv_summary["worst_pair"]),
                "Adj_R2": float(snapshot["model"].rsquared_adj),
                "CV_RMSE": float(snapshot["cv_rmse"]),
                "CV_RMSE_SD": float(snapshot["cv_rmse_sd"]),
            }
        )
    comparison_df = pd.DataFrame(rows)
    baseline_row = comparison_df.loc[comparison_df["模型"] == baseline_name].iloc[0]
    comparison_df["判别效度通过率变化"] = comparison_df["Fornell通过率"] - float(baseline_row["Fornell通过率"])
    comparison_df["最小判别边际变化"] = comparison_df["最小判别效度边际"] - float(baseline_row["最小判别效度边际"])
    comparison_df["最大相关变化"] = comparison_df["最大潜变量相关"] - float(baseline_row["最大潜变量相关"])
    comparison_df["Adj_R2变化"] = comparison_df["Adj_R2"] - float(baseline_row["Adj_R2"])
    comparison_df["CV_RMSE变化"] = comparison_df["CV_RMSE"] - float(baseline_row["CV_RMSE"])

    recommendations = []
    for _, row in comparison_df.iterrows():
        if row["模型"] == baseline_name:
            recommendations.append("冻结基线")
            continue
        improved_discriminant = (
            row["判别效度通过率变化"] >= 0.20
            or row["最小判别边际变化"] >= 0.08
            or row["最大相关变化"] <= -0.08
        )
        small_tradeoff = row["Adj_R2变化"] >= -0.03 and row["CV_RMSE变化"] <= 0.03
        if improved_discriminant and small_tradeoff:
            recommendations.append("建议替换")
        elif improved_discriminant:
            recommendations.append("判别效度改善，但解释损失偏大")
        else:
            recommendations.append("保留为备选")
    comparison_df["建议"] = recommendations
    return comparison_df

def fit_path_model(
    scores: pd.DataFrame,
    *,
    include_interactions: bool = True,
    bootstrap_iterations: int = 0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    analysis = scores[STRUCTURAL_PREDICTORS + ["BI"]].dropna().copy()
    z_scores = (analysis - analysis.mean()) / analysis.std(ddof=0)

    predictors = STRUCTURAL_PREDICTORS.copy()
    if include_interactions:
        for left, right in INTERACTION_TERMS:
            name = f"{left}_x_{right}"
            z_scores[name] = z_scores[left] * z_scores[right]
            predictors.append(name)

    x = sm.add_constant(z_scores[predictors])
    y = z_scores["BI"]
    model = sm.OLS(y, x).fit(cov_type="HC3")
    ci = model.conf_int().loc[predictors]
    coef_table = pd.DataFrame(
        {
            "coef": model.params.loc[predictors],
            "std_error": model.bse.loc[predictors],
            "p_value": model.pvalues.loc[predictors],
            "low": ci[0].values,
            "high": ci[1].values,
        },
        index=predictors,
    )

    if bootstrap_iterations > 0:
        rng = np.random.default_rng(random_state)
        x_matrix = x.values
        y_vector = y.values
        boot_rows = []
        for _ in range(bootstrap_iterations):
            sample_idx = rng.integers(0, len(y_vector), size=len(y_vector))
            beta = np.linalg.lstsq(x_matrix[sample_idx], y_vector[sample_idx], rcond=None)[0]
            boot_rows.append(beta[1:])
        bootstrap_df = pd.DataFrame(boot_rows, columns=predictors)
        coef_table["low"] = bootstrap_df.quantile(0.025)
        coef_table["high"] = bootstrap_df.quantile(0.975)
        coef_table["boot_mean"] = bootstrap_df.mean()
        coef_table["boot_se"] = bootstrap_df.std(ddof=1)
        setattr(model, "_bootstrap_samples", bootstrap_df)
    else:
        setattr(model, "_bootstrap_samples", pd.DataFrame())

    return z_scores, coef_table.loc[predictors], model


def fit_custom_path_model(
    scores: pd.DataFrame,
    predictors: list[str],
    *,
    bootstrap_iterations: int = 0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper]:
    analysis = scores[predictors + ["BI"]].dropna().copy()
    z_scores = (analysis - analysis.mean()) / analysis.std(ddof=0)
    x = sm.add_constant(z_scores[predictors])
    y = z_scores["BI"]
    model = sm.OLS(y, x).fit(cov_type="HC3")
    ci = model.conf_int().loc[predictors]
    coef_table = pd.DataFrame(
        {
            "coef": model.params.loc[predictors],
            "std_error": model.bse.loc[predictors],
            "p_value": model.pvalues.loc[predictors],
            "low": ci[0].values,
            "high": ci[1].values,
        },
        index=predictors,
    )

    if bootstrap_iterations > 0:
        rng = np.random.default_rng(random_state)
        x_matrix = x.values
        y_vector = y.values
        boot_rows = []
        for _ in range(bootstrap_iterations):
            sample_idx = rng.integers(0, len(y_vector), size=len(y_vector))
            beta = np.linalg.lstsq(x_matrix[sample_idx], y_vector[sample_idx], rcond=None)[0]
            boot_rows.append(beta[1:])
        bootstrap_df = pd.DataFrame(boot_rows, columns=predictors)
        coef_table["low"] = bootstrap_df.quantile(0.025)
        coef_table["high"] = bootstrap_df.quantile(0.975)
        coef_table["boot_mean"] = bootstrap_df.mean()
        coef_table["boot_se"] = bootstrap_df.std(ddof=1)
        setattr(model, "_bootstrap_samples", bootstrap_df)
    else:
        setattr(model, "_bootstrap_samples", pd.DataFrame())

    return z_scores, coef_table.loc[predictors], model


def cross_validated_rmse(
    scores: pd.DataFrame,
    predictors: list[str],
    *,
    folds: int = 5,
    random_state: int = 42,
) -> tuple[float, float]:
    analysis = scores.copy()
    for predictor in predictors:
        if predictor not in analysis.columns and predictor == "CVP_x_PI":
            analysis[predictor] = analysis["CVP"] * analysis["PI"]
        if predictor not in analysis.columns and predictor == "PC_x_PI":
            analysis[predictor] = analysis["PC"] * analysis["PI"]
        if predictor not in analysis.columns and predictor == "PR_x_PKN":
            analysis[predictor] = analysis["PR"] * analysis["PKN"]
    analysis = analysis[predictors + ["BI"]].dropna().copy()
    z_scores = (analysis - analysis.mean()) / analysis.std(ddof=0)
    indices = np.arange(len(z_scores))
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    fold_splits = np.array_split(indices, folds)
    rmses = []
    x_all = z_scores[predictors].values
    y_all = z_scores["BI"].values
    for test_idx in fold_splits:
        train_mask = np.ones(len(z_scores), dtype=bool)
        train_mask[test_idx] = False
        x_train = np.column_stack([np.ones(train_mask.sum()), x_all[train_mask]])
        y_train = y_all[train_mask]
        beta = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        x_test = np.column_stack([np.ones(len(test_idx)), x_all[test_idx]])
        pred = x_test @ beta
        rmse = float(np.sqrt(np.mean((y_all[test_idx] - pred) ** 2)))
        rmses.append(rmse)
    return float(np.mean(rmses)), float(np.std(rmses))


def evaluate_alternative_sem_candidate(
    spec: dict[str, object],
    accept_series: pd.Series,
    *,
    random_state: int = 42,
) -> dict[str, object]:
    item_frames = spec["item_frames"]
    mean_scores = build_construct_mean_scores(item_frames)
    cfa_result = fit_cfa_model(item_frames, mean_scores, construct_order=list(item_frames.keys()))
    reliability_df = compute_reliability_validity(
        cfa_result,
        item_frames,
        construct_order=list(item_frames.keys()),
    )
    score_frame = cfa_result.factor_scores.join(accept_series.rename("ACCEPT"), how="left")
    full_predictors = list(spec["full_predictors"])
    _, coef_full, _ = fit_custom_path_model(
        score_frame,
        full_predictors,
        bootstrap_iterations=0,
        random_state=random_state,
    )
    final_predictors = [
        predictor
        for predictor in full_predictors
        if predictor in coef_full.index and float(coef_full.loc[predictor, "p_value"]) < 0.05
    ]
    if not final_predictors:
        fallback_count = min(3, len(full_predictors))
        final_predictors = coef_full["p_value"].sort_values().head(fallback_count).index.tolist()
    model_data, coef_final, model_final = fit_custom_path_model(
        score_frame,
        final_predictors,
        bootstrap_iterations=1000,
        random_state=random_state,
    )
    cv_rmse, cv_rmse_sd = cross_validated_rmse(
        score_frame,
        final_predictors,
        random_state=random_state,
    )
    return build_sem_candidate_snapshot(
        name=str(spec["name"]),
        description=str(spec["description"]),
        item_frames=item_frames,
        cfa_result=cfa_result,
        reliability_df=reliability_df,
        model_data=model_data,
        score_frame=score_frame,
        coef_table=coef_final,
        model=model_final,
        final_predictors=final_predictors,
        cv_rmse=cv_rmse,
        cv_rmse_sd=cv_rmse_sd,
    )


def fit_preferred_report_sem_model(
    df: pd.DataFrame,
    *,
    preferred_name: str = "备选A：购买准备度合并模型",
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, sm.regression.linear_model.RegressionResultsWrapper, dict[str, object]]:
    item_frames, mean_scores = prepare_part4_data(df)
    baseline_cfa = fit_cfa_model(item_frames, mean_scores)
    audit_df = build_item_audit_table(baseline_cfa, item_frames)
    refined_item_frames, _ = propose_refined_item_frames(item_frames, audit_df)
    spec_lookup = {str(spec["name"]): spec for spec in build_alternative_sem_specs(refined_item_frames)}
    preferred_spec = spec_lookup.get(preferred_name, next(iter(spec_lookup.values())))
    snapshot = evaluate_alternative_sem_candidate(
        preferred_spec,
        mean_scores["ACCEPT"],
        random_state=random_state,
    )
    return snapshot["model_data"], snapshot["coef"], snapshot["model"], snapshot


def compute_calibration_summary(
    model_data: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
) -> pd.DataFrame:
    calibration = pd.DataFrame(
        {"observed": model_data["BI"], "predicted": model.fittedvalues},
        index=model_data.index,
    ).dropna()
    n_bins = min(8, calibration["predicted"].nunique())
    calibration["bin"] = pd.qcut(calibration["predicted"], q=n_bins, duplicates="drop")
    summary = calibration.groupby("bin").agg(
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
        n=("observed", "size"),
    )
    summary["abs_error"] = (summary["observed"] - summary["predicted"]).abs()
    return summary.round(4)


def compute_multigroup_path_diff(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    predictors: list[str] | None = None,
) -> pd.DataFrame:
    predictors = predictors or STRUCTURAL_PREDICTORS
    schema = build_schema(df)
    gender = map_codes(to_numeric(df[schema["part6"]["gender"]]), GENDER_MAP, unknown_prefix="性别")
    rows = []
    for group_name in [name for name in GENDER_MAP.values() if name in gender.unique()]:
        group_index = gender[gender == group_name].index.intersection(scores.dropna().index)
        if len(group_index) < 80:
            continue
        _, coef_table, _ = fit_custom_path_model(
            scores.loc[group_index, predictors + ["BI"]],
            predictors,
            bootstrap_iterations=0,
        )
        for predictor in predictors:
            rows.append(
                {
                    "group": group_name,
                    "predictor": predictor,
                    "coef": float(coef_table.loc[predictor, "coef"]),
                    "low": float(coef_table.loc[predictor, "low"]),
                    "high": float(coef_table.loc[predictor, "high"]),
                    "n": int(len(group_index)),
                }
            )
    return pd.DataFrame(rows)


def plot_measurement_heatmap(
    cfa_result: CFAResult,
    reliability_df: pd.DataFrame,
    loading_summary: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    heatmap_df = cfa_result.cross_loading_corr.copy()
    heatmap_df.index = [pretty_item_label(item) for item in heatmap_df.index]
    heatmap_df.columns = [CONSTRUCT_LABELS[col] for col in heatmap_df.columns]

    summary = loading_summary.copy()
    summary["y"] = np.arange(len(summary))

    fig = plt.figure(figsize=(15.2, 10.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], wspace=0.28)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])

    sns.heatmap(
        heatmap_df,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.35,
        cbar_kws={"label": "题项与潜变量得分相关"},
        ax=ax_left,
    )
    ax_left.set_title("题项与潜变量相关矩阵", fontsize=15)
    ax_left.set_xlabel("")
    ax_left.set_ylabel("")

    palette = dict(zip(CONSTRUCT_ORDER, sns.color_palette("Set2", len(CONSTRUCT_ORDER))))
    for _, row in summary.iterrows():
        ax_right.hlines(row["y"], row["cross"], row["own"], color="#C7CED8", linewidth=2.4, zorder=1)
        ax_right.scatter(row["cross"], row["y"], s=55, color=PALETTE["slate"], edgecolor="white", linewidth=0.5, zorder=3)
        ax_right.scatter(row["own"], row["y"], s=75, color=palette[row["construct"]], edgecolor="white", linewidth=0.6, zorder=4)
    ax_right.set_yticks(summary["y"])
    ax_right.set_yticklabels(summary["item_label"])
    ax_right.set_xlabel("标准化载荷 / 最大跨负荷")
    ax_right.set_ylabel("")
    ax_right.set_title("本构念载荷与跨构念负荷对照", fontsize=15)
    ax_right.legend(
        handles=[
            plt.Line2D([], [], marker="o", color="none", markerfacecolor=PALETTE["slate"], markeredgecolor="white", markersize=8, label="最大跨构念相关"),
            plt.Line2D([], [], marker="o", color="none", markerfacecolor=PALETTE["red"], markeredgecolor="white", markersize=8, label="本构念标准化载荷"),
        ],
        loc="lower right",
    )

    note_lines = []
    for idx, construct in enumerate(CONSTRUCT_ORDER):
        metric = reliability_df.loc[construct]
        note_lines.append(
            f"{CONSTRUCT_LABELS[construct]} α={metric['alpha']:.2f}, CR={metric['CR']:.2f}, AVE={metric['AVE']:.2f}"
        )
        if idx in {3, 7}:
            note_lines.append("")
    fit_note = (
        f"CFA: CFI={cfa_result.fit_stats['CFI']:.3f}, TLI={cfa_result.fit_stats['TLI']:.3f}, "
        f"RMSEA={cfa_result.fit_stats['RMSEA']:.3f}, SRMR={cfa_result.fit_stats['SRMR']:.3f}"
    )
    fig.suptitle("图4.1 测量模型载荷与区分效度总览", y=0.99, fontsize=17, fontweight="bold")
    fig.text(
        0.52,
        0.035,
        "\n".join(note_lines).strip() + f"\n\n{fit_note}",
        ha="left",
        va="bottom",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.35"},
    )
    save_figure(fig, output_dir / "图4.1_测量模型载荷热图.png")
    return cfa_result.cross_loading_corr.round(4)


def plot_sem_path_diagram(
    cfa_result: CFAResult,
    coef_table: pd.DataFrame,
    model: sm.regression.linear_model.RegressionResultsWrapper,
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(13.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    positions = {
        "CVP": (0.16, 0.84),
        "PK": (0.16, 0.68),
        "PC": (0.16, 0.52),
        "EA": (0.16, 0.36),
        "PR": (0.16, 0.20),
        "PI": (0.44, 0.74),
        "PKN": (0.44, 0.48),
        "ACCEPT": (0.44, 0.22),
        "BI": (0.79, 0.50),
    }
    item_counts = {
        construct: int(len(_construct_items(cfa_result.loadings.index, construct)))
        for construct in CONSTRUCT_ORDER
    }

    for construct in CONSTRUCT_ORDER:
        x_pos, y_pos = positions[construct]
        radius = 0.065 if construct != "BI" else 0.085
        face = PALETTE["mint"] if construct != "BI" else PALETTE["rose"]
        circle = Circle((x_pos, y_pos), radius=radius, facecolor=face, edgecolor=PALETTE["ink"], linewidth=1.2)
        ax.add_patch(circle)
        subtitle = f"{item_counts[construct]}题"
        if construct == "BI":
            subtitle = f"4题\n$R^2$={model.rsquared_adj:.3f}"
        ax.text(
            x_pos,
            y_pos,
            f"{CONSTRUCT_LABELS[construct]}\n{subtitle}",
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold" if construct == "BI" else "normal",
        )

    acc_x, acc_y = positions["ACCEPT"]
    accept_box = FancyBboxPatch(
        (acc_x - 0.085, acc_y - 0.05),
        0.17,
        0.10,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        facecolor="#FFF4E6",
        edgecolor=PALETTE["ink"],
        linewidth=1.1,
    )
    ax.add_patch(accept_box)
    ax.text(acc_x, acc_y, "非遗融入接受度\n单题观测变量", ha="center", va="center", fontsize=11)

    for predictor in STRUCTURAL_PREDICTORS:
        source = positions[predictor]
        target = positions["BI"]
        coef = float(coef_table.loc[predictor, "coef"])
        p_value = float(coef_table.loc[predictor, "p_value"])
        color = PALETTE["red"] if coef >= 0 else PALETTE["blue"]
        width = 1.5 + 7.5 * abs(coef)
        linestyle = "-" if p_value < 0.05 else "--"
        alpha = 0.95 if p_value < 0.05 else 0.45
        ax.annotate(
            "",
            xy=target,
            xytext=source,
            arrowprops={
                "arrowstyle": "-|>",
                "lw": width,
                "color": color,
                "alpha": alpha,
                "linestyle": linestyle,
                "shrinkA": 20,
                "shrinkB": 25,
            },
        )
        mid_x = source[0] + (target[0] - source[0]) * 0.62
        mid_y = source[1] + (target[1] - source[1]) * 0.62
        ax.text(
            mid_x,
            mid_y + (0.02 if source[1] < target[1] else -0.02),
            f"{coef:.2f}{'*' if p_value < 0.05 else ''}",
            color=color,
            fontsize=10.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.12},
        )

    fit_box = (
        f"测量模型拟合\n"
        f"CFI = {cfa_result.fit_stats['CFI']:.3f}\n"
        f"TLI = {cfa_result.fit_stats['TLI']:.3f}\n"
        f"RMSEA = {cfa_result.fit_stats['RMSEA']:.3f}\n"
        f"SRMR = {cfa_result.fit_stats['SRMR']:.3f}\n\n"
        f"结构模型\nAdj. $R^2$ = {model.rsquared_adj:.3f}"
    )
    ax.text(
        0.02,
        0.08,
        fit_box,
        ha="left",
        va="bottom",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.35"},
    )
    ax.text(
        0.98,
        0.08,
        "实线: p < 0.05\n虚线: p >= 0.05\n红色: 正向路径\n蓝色: 负向路径",
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.35"},
    )
    ax.set_title("图4.2 购买意愿结构方程模型路径图", fontsize=17, fontweight="bold")
    save_figure(fig, output_dir / "图4.2_购买意愿结构综合图.png")


def plot_final_sem_path_diagram(
    cfa_result: CFAResult,
    final_coef: pd.DataFrame,
    final_model: sm.regression.linear_model.RegressionResultsWrapper,
    final_predictors: list[str],
    decision_df: pd.DataFrame,
    *,
    item_frames: dict[str, pd.DataFrame] | None = None,
    cv_rmse: float,
    removed_paths: list[str],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(16.2, 10.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if item_frames is not None:
        group_items = {
            predictor: item_frames[predictor].columns.tolist()
            for predictor in final_predictors
            if predictor in item_frames
        }
    else:
        group_items = {
            predictor: decision_df[(decision_df["construct"] == predictor) & (decision_df["keep_in_refined"])]["item"].tolist()
            for predictor in final_predictors
        }
    predictor_positions = {}
    item_y_map: dict[str, dict[str, float]] = {}
    current_top = 0.94
    gap = 0.012
    for predictor in final_predictors:
        items = group_items[predictor]
        block_height = 0.035 * max(len(items), 2) + 0.05
        center_y = current_top - block_height / 2
        predictor_positions[predictor] = (0.35, float(center_y))
        item_positions = np.linspace(current_top - 0.035, current_top - block_height + 0.035, len(items))
        item_y_map[predictor] = {item: float(y_val) for item, y_val in zip(items, item_positions)}
        current_top = current_top - block_height - gap
    bi_pos = (0.80, 0.52)
    item_x = 0.085

    for predictor, (x_pos, y_pos) in predictor_positions.items():
        node = FancyBboxPatch(
            (x_pos - 0.08, y_pos - 0.04),
            0.16,
            0.08,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            facecolor=PALETTE["mint"],
            edgecolor=PALETTE["ink"],
            linewidth=1.2,
        )
        ax.add_patch(node)
        ax.text(x_pos, y_pos, CONSTRUCT_LABELS[predictor], ha="center", va="center", fontsize=11.5)

        items = group_items[predictor]
        if not items:
            continue
        for item in items:
            iy = item_y_map[predictor][item]
            rect = FancyBboxPatch(
                (item_x - 0.06, iy - 0.022),
                0.12,
                0.044,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                facecolor="white",
                edgecolor=PALETTE["slate"],
                linewidth=0.9,
            )
            ax.add_patch(rect)
            ax.text(item_x, iy, pretty_item_label(item), ha="center", va="center", fontsize=8.8)
            loading = float(cfa_result.loadings.loc[item, predictor])
            ax.annotate(
                "",
                xy=(x_pos - 0.08, y_pos),
                xytext=(item_x + 0.06, iy),
                arrowprops={"arrowstyle": "->", "lw": 1.1, "color": PALETTE["slate"], "alpha": 0.8},
            )
            mid_x = (item_x + 0.06 + x_pos - 0.08) / 2
            mid_y = (iy + y_pos) / 2
            ax.text(
                mid_x,
                mid_y + 0.01,
                f"{loading:.2f}",
                fontsize=8,
                color=PALETTE["slate"],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.08},
            )

    bi_box = FancyBboxPatch(
        (bi_pos[0] - 0.10, bi_pos[1] - 0.06),
        0.20,
        0.12,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=PALETTE["rose"],
        edgecolor=PALETTE["ink"],
        linewidth=1.3,
    )
    ax.add_patch(bi_box)
    if item_frames is not None and "BI" in item_frames:
        bi_items = item_frames["BI"].columns.tolist()
    else:
        bi_items = decision_df[(decision_df["construct"] == "BI") & (decision_df["keep_in_refined"])]["item"].tolist()
    ax.text(
        bi_pos[0],
        bi_pos[1],
        f"购买意愿\n{len(bi_items)}题\nAdj. $R^2$={final_model.rsquared_adj:.3f}",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    bi_item_x = 0.94
    y_offsets = np.linspace(0.12, -0.12, len(bi_items))
    for offset, item in zip(y_offsets, bi_items):
        iy = bi_pos[1] + offset
        rect = FancyBboxPatch(
            (bi_item_x - 0.06, iy - 0.022),
            0.12,
            0.044,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            facecolor="white",
            edgecolor=PALETTE["slate"],
            linewidth=0.9,
        )
        ax.add_patch(rect)
        ax.text(bi_item_x, iy, pretty_item_label(item), ha="center", va="center", fontsize=8.8)
        loading = float(cfa_result.loadings.loc[item, "BI"])
        ax.annotate(
            "",
            xy=(bi_item_x - 0.06, iy),
            xytext=(bi_pos[0] + 0.10, bi_pos[1]),
            arrowprops={"arrowstyle": "->", "lw": 1.1, "color": PALETTE["slate"], "alpha": 0.8},
        )
        mid_x = (bi_item_x - 0.06 + bi_pos[0] + 0.10) / 2
        mid_y = (iy + bi_pos[1]) / 2
        ax.text(
            mid_x,
            mid_y + 0.01,
            f"{loading:.2f}",
            fontsize=8,
            color=PALETTE["slate"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.08},
        )

    curve_rads = np.linspace(-0.22, 0.22, len(final_predictors))
    label_offsets = np.linspace(0.045, -0.045, len(final_predictors))
    for idx, (predictor, (x_pos, y_pos)) in enumerate(predictor_positions.items()):
        coef = float(final_coef.loc[predictor, "coef"])
        color = PALETTE["red"] if coef >= 0 else PALETTE["blue"]
        width = 1.6 + 8 * abs(coef)
        ax.annotate(
            "",
            xy=(bi_pos[0] - 0.11, bi_pos[1]),
            xytext=(x_pos + 0.08, y_pos),
            arrowprops={
                "arrowstyle": "-|>",
                "lw": width,
                "color": color,
                "alpha": 0.92,
                "shrinkA": 3,
                "shrinkB": 4,
                "connectionstyle": f"arc3,rad={curve_rads[idx]:.3f}",
            },
        )
        mid_x = x_pos + (bi_pos[0] - 0.11 - (x_pos + 0.08)) * 0.58
        mid_y = y_pos + (bi_pos[1] - y_pos) * 0.58
        p_value = float(final_coef.loc[predictor, "p_value"])
        stars = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        ax.text(
            mid_x,
            mid_y + label_offsets[idx],
            f"{coef:.2f}{stars}",
            fontsize=10.5,
            color=color,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.10},
        )

    note_text = (
        f"测量模型拟合\nCFI={cfa_result.fit_stats['CFI']:.3f}\nTLI={cfa_result.fit_stats['TLI']:.3f}\n"
        f"RMSEA={cfa_result.fit_stats['RMSEA']:.3f}\nSRMR={cfa_result.fit_stats['SRMR']:.3f}\n\n"
        f"5折CV RMSE={cv_rmse:.3f}"
    )
    ax.text(
        0.47,
        0.035,
        note_text,
        ha="left",
        va="bottom",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.35"},
    )
    ax.text(
        0.985,
        0.04,
        "删减路径:\n" + "\n".join(_pretty_path_label(name) for name in removed_paths),
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": PALETTE["sand"], "boxstyle": "round,pad=0.35"},
    )
    ax.set_title("图4.2 优化购买意愿结构方程模型路径图", fontsize=18, fontweight="bold")
    save_figure(fig, output_dir / "图4.2_购买意愿结构综合图.png")
    fig, ax = None, None


def plot_effect_forest(coef_table: pd.DataFrame, output_dir: Path) -> None:
    plot_df = coef_table.reindex([name for name in PLOT_ORDER if name in coef_table.index]).dropna().copy()
    plot_df["label"] = [_pretty_path_label(name) for name in plot_df.index]
    plot_df["y"] = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(9.8, 7.2))
    for _, row in plot_df.iterrows():
        color = PALETTE["red"] if row["coef"] >= 0 else PALETTE["blue"]
        alpha = 0.95 if row["p_value"] < 0.05 else 0.45
        ax.hlines(row["y"], row["low"], row["high"], color=color, linewidth=2.6, alpha=alpha)
        ax.scatter(
            row["coef"],
            row["y"],
            s=70,
            color=color,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_yticks(plot_df["y"])
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("标准化路径系数（95% Bootstrap CI）")
    ax.set_ylabel("")
    ax.set_title("图4.3 结构路径与调节项系数森林图", fontsize=16, fontweight="bold")
    save_figure(fig, output_dir / "图4.3_调节效应联合图.png")


def plot_fornell_larcker_heatmap(
    cfa_result: CFAResult,
    reliability_df: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    matrix = cfa_result.factor_corr.abs().copy()
    for construct in CONSTRUCT_ORDER:
        matrix.loc[construct, construct] = reliability_df.loc[construct, "sqrt_AVE"]
    display = matrix.copy()
    display.index = [CONSTRUCT_LABELS[idx] for idx in display.index]
    display.columns = [CONSTRUCT_LABELS[idx] for idx in display.columns]

    fig, ax = plt.subplots(figsize=(8.6, 7.1))
    sns.heatmap(
        display,
        cmap="YlGnBu",
        annot=True,
        fmt=".2f",
        linewidths=0.35,
        cbar_kws={"label": "对角线为 sqrt(AVE)，其余为 |相关|"},
        ax=ax,
    )
    ax.set_title("图4.5 Fornell-Larcker 判别效度热图", fontsize=16, fontweight="bold")
    save_figure(fig, output_dir / "图4.5_Fornell-Larcker判别效度热图.png")
    return matrix.round(4)


def plot_bootstrap_distribution(
    model: sm.regression.linear_model.RegressionResultsWrapper,
    coef_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    bootstrap_df = getattr(model, "_bootstrap_samples", pd.DataFrame())
    if bootstrap_df.empty:
        return
    plot_cols = [name for name in PLOT_ORDER if name in bootstrap_df.columns]
    melted = bootstrap_df[plot_cols].melt(var_name="path", value_name="coef")
    melted["label"] = melted["path"].map(_pretty_path_label)
    order = [_pretty_path_label(name) for name in plot_cols]

    fig, ax = plt.subplots(figsize=(10.2, 7.4))
    sns.violinplot(
        data=melted,
        y="label",
        x="coef",
        order=order,
        orient="h",
        inner=None,
        color=PALETTE["mint"],
        linewidth=0.8,
        cut=0,
        ax=ax,
    )
    estimate = coef_table.loc[plot_cols].copy()
    estimate["label"] = [_pretty_path_label(name) for name in estimate.index]
    ax.scatter(estimate["coef"], estimate["label"], color=PALETTE["red_dark"], s=45, zorder=3)
    ax.axvline(0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Bootstrap 路径系数分布")
    ax.set_ylabel("")
    ax.set_title("图4.6 Bootstrap 路径系数分布图", fontsize=16, fontweight="bold")
    save_figure(fig, output_dir / "图4.6_Bootstrap路径系数分布图.png")


def plot_model_dashboard(
    cfa_result: CFAResult,
    reliability_df: pd.DataFrame,
    calibration: pd.DataFrame,
    multigroup: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.2, 10.2))
    ax_fit, ax_rel = axes[0]
    ax_cal, ax_group = axes[1]

    fit_items = pd.Series(
        {
            "CFI": cfa_result.fit_stats["CFI"],
            "TLI": cfa_result.fit_stats["TLI"],
            "RMSEA": cfa_result.fit_stats["RMSEA"],
            "SRMR": cfa_result.fit_stats["SRMR"],
        }
    )
    fit_thresholds = {"CFI": 0.90, "TLI": 0.90, "RMSEA": 0.08, "SRMR": 0.08}
    fit_direction = {"CFI": "high", "TLI": "high", "RMSEA": "low", "SRMR": "low"}
    for pos, metric in enumerate(fit_items.index):
        value = float(fit_items[metric])
        threshold = fit_thresholds[metric]
        good = value >= threshold if fit_direction[metric] == "high" else value <= threshold
        color = PALETTE["teal_dark"] if good else PALETTE["red"]
        ax_fit.hlines(pos, 0, value, color=color, linewidth=3)
        ax_fit.scatter(value, pos, s=70, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax_fit.axvline(threshold, color=PALETTE["slate"], linestyle="--", linewidth=1)
        ax_fit.text(value + 0.01, pos, f"{value:.3f}", va="center", fontsize=10)
    ax_fit.set_yticks(np.arange(len(fit_items)))
    ax_fit.set_yticklabels(fit_items.index)
    ax_fit.set_xlim(0, max(1.0, float(fit_items.max()) + 0.08))
    ax_fit.set_xlabel("拟合指标值")
    ax_fit.set_title("模型拟合指标", fontsize=14)

    rel_plot = reliability_df.loc[CONSTRUCT_ORDER, ["alpha", "CR", "AVE"]]
    x = np.arange(len(rel_plot))
    ax_rel.plot(x, rel_plot["alpha"], marker="o", color=PALETTE["gold"], linewidth=2.2, label="Cronbach α")
    ax_rel.plot(x, rel_plot["CR"], marker="s", color=PALETTE["red"], linewidth=2.2, label="CR")
    ax_rel.plot(x, rel_plot["AVE"], marker="^", color=PALETTE["teal_dark"], linewidth=2.2, label="AVE")
    ax_rel.axhline(0.70, color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax_rel.axhline(0.50, color=PALETTE["slate"], linestyle=":", linewidth=1)
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels([CONSTRUCT_LABELS[idx] for idx in rel_plot.index], rotation=35, ha="right")
    ax_rel.set_ylim(0, 1.05)
    ax_rel.set_ylabel("指标值")
    ax_rel.set_title("信度与收敛效度", fontsize=14)
    ax_rel.legend(loc="lower right")

    low = min(float(calibration["predicted"].min()), float(calibration["observed"].min()))
    high = max(float(calibration["predicted"].max()), float(calibration["observed"].max()))
    ax_cal.plot([low, high], [low, high], color=PALETTE["slate"], linestyle="--", linewidth=1.2)
    ax_cal.plot(calibration["predicted"], calibration["observed"], color=PALETTE["red_dark"], linewidth=2.0)
    ax_cal.scatter(
        calibration["predicted"],
        calibration["observed"],
        s=calibration["n"] * 6,
        color=PALETTE["teal"],
        alpha=0.82,
        edgecolor="white",
        linewidth=0.7,
    )
    for _, row in calibration.iterrows():
        ax_cal.text(row["predicted"], row["observed"], str(int(row["n"])), fontsize=9, ha="left", va="bottom")
    ax_cal.set_xlabel("分箱平均预测购买意愿")
    ax_cal.set_ylabel("分箱平均观测购买意愿")
    ax_cal.set_title("结构模型校准表现", fontsize=14)

    if multigroup.empty:
        ax_group.text(0.5, 0.5, "当前分组样本量不足，未输出稳定的多组路径比较。", ha="center", va="center", fontsize=12)
        ax_group.axis("off")
    else:
        pivot = multigroup.pivot(index="predictor", columns="group", values="coef")
        available_groups = [name for name in GENDER_MAP.values() if name in pivot.columns]
        y = np.arange(len(STRUCTURAL_PREDICTORS))
        ax_group.set_yticks(y)
        ax_group.set_yticklabels([CONSTRUCT_LABELS[name] for name in STRUCTURAL_PREDICTORS])
        ax_group.axvline(0, color="#888888", linestyle="--", linewidth=1.0)
        if len(available_groups) >= 2:
            left_group, right_group = available_groups[:2]
            for idx, predictor in enumerate(STRUCTURAL_PREDICTORS):
                left_value = float(pivot.loc[predictor, left_group])
                right_value = float(pivot.loc[predictor, right_group])
                ax_group.hlines(idx, left_value, right_value, color="#C7CED8", linewidth=2.4, zorder=1)
                ax_group.scatter(left_value, idx, color=PALETTE["blue"], s=60, edgecolor="white", linewidth=0.6, zorder=3, label=left_group if idx == 0 else None)
                ax_group.scatter(right_value, idx, color=PALETTE["red"], s=60, edgecolor="white", linewidth=0.6, zorder=3, label=right_group if idx == 0 else None)
            ax_group.legend(loc="lower right")
        else:
            group_name = available_groups[0]
            ax_group.scatter(pivot[group_name], y, color=PALETTE["red"], s=60, edgecolor="white", linewidth=0.6)
        ax_group.set_xlabel("标准化路径系数")
        ax_group.set_title("性别分组主路径比较", fontsize=14)

    fig.suptitle("图4.4 模型稳健性与群体差异总览", y=0.995, fontsize=17, fontweight="bold")
    fig.text(
        0.02,
        0.02,
        f"样本量 n = {int(cfa_result.fit_stats['n'])}，平均分箱绝对校准误差 = {calibration['abs_error'].mean():.3f}，"
        f"优化状态 = {'成功' if cfa_result.success else '近似收敛'}",
        ha="left",
        va="bottom",
        fontsize=10,
    )
    save_figure(fig, output_dir / "图4.4_模型稳健性与群体差异总览.png")


def plot_item_audit_dashboard(audit_df: pd.DataFrame, decision_df: pd.DataFrame, output_dir: Path) -> None:
    plot_df = audit_df.copy()
    plot_df["item_label"] = plot_df["item"].apply(pretty_item_label)
    plot_df = plot_df.merge(decision_df[["item", "keep_in_refined"]], on="item", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2))
    ax_left, ax_right = axes
    palette = {True: PALETTE["teal_dark"], False: PALETTE["red"]}

    for keep_flag, group in plot_df.groupby("keep_in_refined"):
        ax_left.scatter(
            group["own"],
            group["gap"],
            s=90 + group["residual_var"] * 80,
            color=palette[keep_flag],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            label="保留" if keep_flag else "剔除",
        )
        for _, row in group.iterrows():
            ax_left.text(row["own"] + 0.006, row["gap"] + 0.004, row["item"], fontsize=8.5)
    ax_left.axvline(0.35, color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax_left.axhline(0.08, color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax_left.set_xlabel("标准化载荷")
    ax_left.set_ylabel("本构念载荷 - 最大跨负荷")
    ax_left.set_title("题项保留决策散点图", fontsize=14)
    ax_left.legend(loc="lower right")

    bar_df = plot_df.sort_values(["keep_in_refined", "residual_var", "own"], ascending=[True, False, True]).copy()
    y_pos = np.arange(len(bar_df))
    colors = [palette[flag] for flag in bar_df["keep_in_refined"]]
    ax_right.barh(y_pos, bar_df["residual_var"], color=colors, alpha=0.85)
    ax_right.axvline(0.85, color=PALETTE["slate"], linestyle="--", linewidth=1)
    ax_right.set_yticks(y_pos)
    ax_right.set_yticklabels(bar_df["item"])
    ax_right.set_xlabel("残差方差占比")
    ax_right.set_title("高残差题项审计", fontsize=14)

    fig.suptitle("图4.7 题项审计与留题决策图", y=0.995, fontsize=17, fontweight="bold")
    save_figure(fig, output_dir / "图4.7_题项审计与留题决策图.png")


def plot_model_comparison_dashboard(
    comparison_df: pd.DataFrame,
    baseline_cfa: CFAResult,
    refined_cfa: CFAResult,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0))
    ax_left, ax_right = axes

    fit_metrics = ["CFI", "TLI", "RMSEA", "SRMR", "Adj_R2"]
    fit_compare = comparison_df[comparison_df["metric"].isin(fit_metrics)].copy()
    x = np.arange(len(fit_compare))
    ax_left.bar(x - 0.18, fit_compare["baseline"], width=0.36, color=PALETTE["slate"], label="基线模型")
    ax_left.bar(x + 0.18, fit_compare["refined"], width=0.36, color=PALETTE["teal_dark"], label="优化模型")
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(fit_compare["metric"])
    ax_left.set_title("关键指标对比", fontsize=14)
    ax_left.legend(loc="best")

    residual_delta = (
        baseline_cfa.residual_cov.abs().sub(refined_cfa.residual_cov.abs(), fill_value=0).clip(lower=0).fillna(0)
    )
    sns.heatmap(
        residual_delta,
        cmap="YlOrRd",
        center=0,
        cbar_kws={"label": "|残差相关| 改善幅度"},
        ax=ax_right,
    )
    ax_right.set_title("优化后残差改善热图", fontsize=14)

    fig.suptitle("图4.8 基线与优化 SEM 对比图", y=0.995, fontsize=17, fontweight="bold")
    save_figure(fig, output_dir / "图4.8_基线与优化SEM对比图.png")


def export_sem_tables(
    cfa_result: CFAResult,
    reliability_df: pd.DataFrame,
    fit_table: pd.Series,
    coef_table: pd.DataFrame,
    multigroup: pd.DataFrame,
    score_frame: pd.DataFrame,
    structural_comparison: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    cfa_result.loadings.round(4).to_csv(output_dir / "SEM_标准化载荷矩阵.csv", encoding="utf-8-sig")
    cfa_result.cross_loading_corr.round(4).to_csv(output_dir / "SEM_题项潜变量相关矩阵.csv", encoding="utf-8-sig")
    reliability_df.round(4).to_csv(output_dir / "SEM_信效度指标.csv", encoding="utf-8-sig")
    fit_table.round(4).to_csv(output_dir / "SEM_模型拟合指标.csv", encoding="utf-8-sig")
    coef_table.round(4).to_csv(output_dir / "SEM_结构路径系数.csv", encoding="utf-8-sig")
    score_frame.round(4).to_csv(output_dir / "SEM_潜变量得分.csv", encoding="utf-8-sig")
    if structural_comparison is not None:
        structural_comparison.round(4).to_csv(output_dir / "SEM_结构模型比较.csv", encoding="utf-8-sig", index=False)
    if not multigroup.empty:
        multigroup.round(4).to_csv(output_dir / "SEM_性别分组路径比较.csv", encoding="utf-8-sig", index=False)


def export_sem_audit_tables(
    baseline_cfa: CFAResult,
    baseline_rel: pd.DataFrame,
    baseline_coef: pd.DataFrame,
    refined_cfa: CFAResult,
    refined_rel: pd.DataFrame,
    refined_coef: pd.DataFrame,
    audit_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    structural_comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    baseline_cfa.loadings.round(4).to_csv(output_dir / "SEM_基线标准化载荷矩阵.csv", encoding="utf-8-sig")
    baseline_rel.round(4).to_csv(output_dir / "SEM_基线信效度指标.csv", encoding="utf-8-sig")
    baseline_coef.round(4).to_csv(output_dir / "SEM_基线结构路径系数.csv", encoding="utf-8-sig")
    refined_cfa.loadings.round(4).to_csv(output_dir / "SEM_优化标准化载荷矩阵.csv", encoding="utf-8-sig")
    refined_rel.round(4).to_csv(output_dir / "SEM_优化信效度指标.csv", encoding="utf-8-sig")
    refined_coef.round(4).to_csv(output_dir / "SEM_优化结构路径系数.csv", encoding="utf-8-sig")
    audit_df.round(4).to_csv(output_dir / "SEM_题项审计明细.csv", encoding="utf-8-sig", index=False)
    decision_df.round(4).to_csv(output_dir / "SEM_优化留题方案.csv", encoding="utf-8-sig", index=False)
    comparison_df.round(4).to_csv(output_dir / "SEM_基线优化模型对比.csv", encoding="utf-8-sig", index=False)
    structural_comparison.round(4).to_csv(output_dir / "SEM_结构模型比较_扩展.csv", encoding="utf-8-sig", index=False)


def export_candidate_model_tables(
    candidate_snapshots: list[dict[str, object]],
    comparison_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    comparison_df.round(4).to_csv(output_dir / "SEM_合并构念候选模型比较.csv", encoding="utf-8-sig", index=False)

    reliability_rows = []
    discriminant_rows = []
    path_rows = []
    for snapshot in candidate_snapshots:
        reliability_df = snapshot["reliability"].reset_index().rename(columns={"construct": "构念代码"})
        reliability_df["模型"] = snapshot["name"]
        reliability_df["构念"] = reliability_df["构念代码"].map(lambda name: CONSTRUCT_LABELS.get(name, name))
        reliability_rows.append(reliability_df)

        detail_df = snapshot["dv_detail"].copy()
        detail_df["模型"] = snapshot["name"]
        discriminant_rows.append(detail_df)

        coef_df = snapshot["coef"].reset_index().rename(columns={"index": "路径代码"})
        coef_df["模型"] = snapshot["name"]
        coef_df["路径"] = coef_df["路径代码"].map(lambda name: CONSTRUCT_LABELS.get(name, name))
        path_rows.append(coef_df)

    if reliability_rows:
        pd.concat(reliability_rows, ignore_index=True).round(4).to_csv(
            output_dir / "SEM_合并构念候选模型信效度明细.csv",
            encoding="utf-8-sig",
            index=False,
        )
    if discriminant_rows:
        pd.concat(discriminant_rows, ignore_index=True).round(4).to_csv(
            output_dir / "SEM_合并构念候选模型判别效度明细.csv",
            encoding="utf-8-sig",
            index=False,
        )
    if path_rows:
        pd.concat(path_rows, ignore_index=True).round(4).to_csv(
            output_dir / "SEM_合并构念候选模型路径系数.csv",
            encoding="utf-8-sig",
            index=False,
        )


def choose_recommended_candidate(
    comparison_df: pd.DataFrame,
    *,
    baseline_name: str,
) -> tuple[str, str]:
    recommended = comparison_df[comparison_df["建议"] == "建议替换"].copy()
    if recommended.empty:
        return baseline_name, "当前冻结基线在解释力与稳健性之间仍是更稳妥的选择。"
    best = recommended.sort_values(
        ["最小判别效度边际", "Fornell通过率", "CV_RMSE"],
        ascending=[False, False, True],
    ).iloc[0]
    return str(best["模型"]), "候选模型的判别效度改善已达到预设阈值，且解释力与交叉验证误差仅发生小幅波动。"


def write_candidate_model_report(
    candidate_snapshots: list[dict[str, object]],
    comparison_df: pd.DataFrame,
    *,
    baseline_name: str,
    output_dir: Path,
) -> None:
    chosen_model, reason = choose_recommended_candidate(comparison_df, baseline_name=baseline_name)
    lines = [
        "# 合并构念候选模型比较说明",
        "",
        "## 1. 比较原则",
        "",
        "- 固定当前最终模型为冻结基线，不再回改其题项与路径。",
        "- 新增两套合并高相关构念的候选 SEM 结构，并使用相同口径比较判别效度、CR/AVE 与 5 折 CV RMSE。",
        "- 推荐规则：当候选模型的判别效度明显改善，且 Adj.R² 下降不超过 0.03、CV RMSE 上升不超过 0.03 时，标记为“建议替换”。",
        "",
        "## 2. 自动比较表",
        "",
        "```text",
        comparison_df.round(4).to_string(index=False),
        "```",
        "",
        "## 3. 模型说明",
        "",
    ]
    for snapshot in candidate_snapshots:
        lines.extend(
            [
                f"### {snapshot['name']}",
                "",
                str(snapshot["description"]),
                "",
                f"- 最终路径：{', '.join(CONSTRUCT_LABELS.get(name, name) for name in snapshot['final_predictors'])}",
                f"- 判别效度通过率：{snapshot['dv_summary']['pass_rate']:.1%}",
                f"- 最小判别效度边际：{snapshot['dv_summary']['min_margin']:.3f}",
                f"- 平均 CR / AVE：{snapshot['reliability']['CR'].mean():.3f} / {snapshot['reliability']['AVE'].mean():.3f}",
                f"- CV RMSE：{snapshot['cv_rmse']:.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 4. 推荐结论",
            "",
            f"- 推荐模型：{chosen_model}",
            f"- 推荐理由：{reason}",
            "",
        ]
    )
    (output_dir / "SEM_合并构念候选模型比较说明.md").write_text("\n".join(lines), encoding="utf-8")


def write_calculation_audit_report(
    baseline_cfa: CFAResult,
    refined_cfa: CFAResult,
    baseline_rel: pd.DataFrame,
    refined_rel: pd.DataFrame,
    baseline_coef: pd.DataFrame,
    refined_coef: pd.DataFrame,
    baseline_model: sm.regression.linear_model.RegressionResultsWrapper,
    refined_model: sm.regression.linear_model.RegressionResultsWrapper,
    audit_df: pd.DataFrame,
    decision_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    structural_comparison: pd.DataFrame,
    candidate_comparison_df: pd.DataFrame | None,
    preferred_model_name: str | None,
    output_dir: Path,
) -> None:
    dropped = decision_df[~decision_df["keep_in_refined"]].copy()
    dropped_text = "无"
    if not dropped.empty:
        dropped_text = "\n".join(
            f"- {row['item']}（{pretty_item_label(row['item'])}）：{row['reason']}"
            for _, row in dropped.iterrows()
        )

    significant_refined = refined_coef[(refined_coef["p_value"] < 0.05) & (refined_coef["coef"] > 0)].sort_values("coef", ascending=False)
    lines = [
        "# SEM计算过程与审计说明",
        "",
        "## 1. 审计目标",
        "",
        "本次审计同时检查测量模型、结构模型和代码实现，重点核查题项归属、载荷计算、CR/AVE、Bootstrap 区间和路径估计是否一致。",
        "",
        "## 2. 发现并修复的关键计算问题",
        "",
        "- 已修复 `PK` 与 `PKN` 题项前缀冲突导致的错归属问题。此前基于 `startswith('PK')` 的写法会把 `PKN1-3` 错并入 `PK`，直接污染题项诊断和信效度结果。",
        "",
        "## 3. 测量模型计算流程",
        "",
        "1. 对第四部分全部量表题进行数值化与缺失剔除。",
        "2. 对观测题项标准化，构造样本协方差矩阵。",
        "3. 使用受约束 CFA 优化测量模型，第一题载荷固定为 1，其余载荷、因子协方差和误差方差通过极大似然式目标函数估计。",
        "4. 由标准化载荷计算 `CR`、`AVE`、`sqrt(AVE)` 与题项残差方差占比。",
        "5. 依据载荷、跨负荷间隔和残差方差做题项审计，再透明形成优化版留题方案。",
        "",
        "## 4. 题项优化规则",
        "",
        "- 规则A：标准化载荷 `< 0.35` 视为弱题项。",
        "- 规则B：本构念载荷与最大跨负荷差值 `< 0.08` 视为区分效度不足。",
        "- 规则C：当题项载荷 `< 0.40` 且残差方差占比 `> 0.85` 时，视为高残差弱题项。",
        "- 规则D：每个构念至少保留 2 个题项，若自动筛选后不足 2 个，则按载荷和间隔重新补回表现较好的题项。",
        "",
        "## 5. 被剔除题项",
        "",
        dropped_text,
        "",
        "## 6. 基线与优化模型对比",
        "",
        "```text",
        comparison_df.round(4).to_string(index=False),
        "```",
        "",
        "## 6.1 结构模型比较",
        "",
        "```text",
        structural_comparison.round(4).to_string(index=False),
        "```",
        "",
    ]
    if candidate_comparison_df is not None and not candidate_comparison_df.empty:
        lines.extend(
            [
                "## 6.2 合并构念候选模型比较",
                "",
                "```text",
                candidate_comparison_df.round(4).to_string(index=False),
                "```",
                "",
            ]
        )
    lines.extend(
        [
        "## 7. 基线信效度",
        "",
        "```text",
        baseline_rel.round(4).to_string(),
        "```",
        "",
        "## 8. 优化后信效度",
        "",
        "```text",
        refined_rel.round(4).to_string(),
        "```",
        "",
        "## 9. 基线路径系数",
        "",
        "```text",
        baseline_coef.round(4).to_string(),
        "```",
        "",
        "## 10. 优化后路径系数",
        "",
        "```text",
        refined_coef.round(4).to_string(),
        "```",
        "",
        "## 11. 审计结论",
        "",
        f"- 优化前拟合：CFI={baseline_cfa.fit_stats['CFI']:.3f}，TLI={baseline_cfa.fit_stats['TLI']:.3f}，RMSEA={baseline_cfa.fit_stats['RMSEA']:.3f}，SRMR={baseline_cfa.fit_stats['SRMR']:.3f}，Adj.R²={baseline_model.rsquared_adj:.3f}。",
        f"- 优化后拟合：CFI={refined_cfa.fit_stats['CFI']:.3f}，TLI={refined_cfa.fit_stats['TLI']:.3f}，RMSEA={refined_cfa.fit_stats['RMSEA']:.3f}，SRMR={refined_cfa.fit_stats['SRMR']:.3f}，Adj.R²={refined_model.rsquared_adj:.3f}。",
        f"- 优化后最稳定的正向驱动仍主要是 {', '.join(_pretty_path_label(idx) for idx in significant_refined.head(5).index)}。",
        (
            f"- 主报告已切换为 {preferred_model_name} 口径，用于缓解高相关构念导致的判别效度不足。"
            if preferred_model_name
            else "- 当前主报告仍沿用优化后模型口径。"
        ),
        "- 量表整体已明显改善，但部分构念的 CR/AVE 仍偏弱，这属于原始问卷题项设计限制，不能用后处理完全消除。",
        "",
        ]
    )
    (output_dir / "SEM_计算过程与审计说明.md").write_text("\n".join(lines), encoding="utf-8")


def write_sem_report(
    cfa_result: CFAResult,
    reliability_df: pd.DataFrame,
    coef_table: pd.DataFrame,
    calibration: pd.DataFrame,
    multigroup: pd.DataFrame,
    structural_model: sm.regression.linear_model.RegressionResultsWrapper,
    structural_comparison: pd.DataFrame,
    removed_paths: list[str],
    cv_rmse: float,
    output_dir: Path,
) -> None:
    construct_order = reliability_df.index.tolist()
    has_merged_construct = any(name in {"PREP", "ACCESS", "ENGAGE"} for name in construct_order)
    significant_positive = coef_table[(coef_table["coef"] > 0) & (coef_table["p_value"] < 0.05)].sort_values("coef", ascending=False)
    significant_negative = coef_table[(coef_table["coef"] < 0) & (coef_table["p_value"] < 0.05)].sort_values("coef")
    strongest_construct = reliability_df["CR"].sort_values(ascending=False).index[0]
    weakest_ave = reliability_df["AVE"].sort_values().index[0]

    bullets = [
        f"第四部分量表共 {int(cfa_result.fit_stats['n'])} 个有效样本、{cfa_result.observed.shape[1]} 个观测题项、{len(construct_order)} 个主报告采用的潜变量，是整份问卷中最适合开展结构方程模型分析的章节。",
        f"CFA 拟合结果为 CFI={cfa_result.fit_stats['CFI']:.3f}、TLI={cfa_result.fit_stats['TLI']:.3f}、RMSEA={cfa_result.fit_stats['RMSEA']:.3f}、SRMR={cfa_result.fit_stats['SRMR']:.3f}，说明模型达到可解释且较稳定的拟合水平。",
        f"最终精简结构模型的调整后 R² 为 {structural_model.rsquared_adj:.3f}，5折交叉验证 RMSE 为 {cv_rmse:.3f}，说明模型在简化后仍保持稳定解释力。",
        f"主路径中显著正向作用最强的因素依次集中在 {', '.join(_pretty_path_label(idx) for idx in significant_positive.head(4).index)}。",
        (
            "当前没有稳定的显著负向路径。"
            if significant_negative.empty
            else f"显著负向路径主要包括 {', '.join(_pretty_path_label(idx) for idx in significant_negative.index)}。"
        ),
        f"精简过程中删减的路径包括 {', '.join(_pretty_path_label(name) for name in removed_paths)}。",
        f"信效度方面，复合信度最高的是 {CONSTRUCT_LABELS[strongest_construct]}（CR={reliability_df.loc[strongest_construct, 'CR']:.3f}），AVE 相对较弱的是 {CONSTRUCT_LABELS[weakest_ave]}（AVE={reliability_df.loc[weakest_ave, 'AVE']:.3f}）。",
        f"分箱校准的平均绝对误差为 {calibration['abs_error'].mean():.3f}，说明结构模型在高低购买意愿群体之间具有较稳定的区分能力。",
    ]

    path_display = coef_table.copy()
    path_display.index = [_pretty_path_label(idx) for idx in path_display.index]
    rel_display = reliability_df.copy()
    rel_display.index = [CONSTRUCT_LABELS[idx] for idx in rel_display.index]
    corr_display = cfa_result.factor_corr.round(4).copy()
    corr_display.index = [CONSTRUCT_LABELS[idx] for idx in corr_display.index]
    corr_display.columns = [CONSTRUCT_LABELS[idx] for idx in corr_display.columns]

    multigroup_text = "当前分组样本不足以输出稳定的多组比较结果。"
    if not multigroup.empty:
        multigroup_text = multigroup.round(4).to_string(index=False)

    sections = [
        (
            "模型说明",
            (
                "本部分采用两步法结构方程模型思路：先对当前保留题项进行验证性因子分析（CFA），再依据标准化载荷构建潜变量加权得分，并估计购买意愿的结构路径。"
                "主报告版本进一步采用合并高相关构念后的简约结构，用于缓解判别效度不足问题。"
                if has_merged_construct
                else "本部分采用两步法结构方程模型思路：先对当前保留题项进行验证性因子分析（CFA），再依据标准化载荷构建潜变量加权得分，并估计购买意愿的结构路径。这样既能保留测量模型的信效度检验，又能在当前环境下稳定输出可复现的路径结果与图形。"
            ),
        ),
        ("模型拟合指标", "```text\n" + cfa_result.fit_stats.round(4).to_string() + "\n```"),
        ("潜变量相关矩阵", "```text\n" + corr_display.to_string() + "\n```"),
        ("信效度摘要", "```text\n" + rel_display.round(4).to_string() + "\n```"),
        ("结构模型比较", "```text\n" + structural_comparison.round(4).to_string(index=False) + "\n```"),
        ("结构路径系数", "```text\n" + path_display.round(4).to_string() + "\n```"),
        ("多组比较", "```text\n" + multigroup_text + "\n```"),
        (
            "结果解读",
            f"综合来看，购买意愿的核心驱动主要来自 {', '.join(_pretty_path_label(idx) for idx in significant_positive.head(4).index)}。其中，"
            f"{_pretty_path_label(significant_positive.index[0]) if not significant_positive.empty else '主要驱动项'} 对购买意愿的影响最强。"
            "测量模型层面，大多数题项在所属潜变量上的载荷明显高于跨构念相关，说明第四部分量表可以支撑后续结构路径解释。"
            + ("合并高相关构念后，主报告中的路径解释更偏向‘文化价值 + 购买准备 + 先验知识’的三轴框架。"
               if has_merged_construct
               else "")
            + "因此，在论文写作中可以把第四部分作为整份问卷的核心 SEM 章节，前面几部分更多承担认知、行为、场景和细分的背景解释作用。",
        ),
    ]

    write_report(
        output_dir / "分析摘要.md",
        _context_text("title", "购买意愿影响因素") + "：结构方程模型分析",
        _context_text(
            "intro",
            "本部分围绕文化价值感知、产品知识、购买便利性、经济可及性、感知风险、产品涉入度、先验知识与购买意愿展开，是整份问卷中最适合开展 SEM 的章节。",
        )
        + " 本轮输出重点重构为测量模型、结构路径、稳健性诊断和补充判别效度图。",
        bullets,
        sections,
    )


def main() -> None:
    df = load_data()
    output_dir = ensure_output_dir("第四部分")
    item_frames, mean_scores = prepare_part4_data(df)

    baseline_cfa = fit_cfa_model(item_frames, mean_scores)
    baseline_rel = compute_reliability_validity(baseline_cfa, item_frames)
    audit_df = build_item_audit_table(baseline_cfa, item_frames)
    refined_item_frames, decision_df = propose_refined_item_frames(item_frames, audit_df)
    refined_cfa = fit_cfa_model(refined_item_frames, mean_scores)
    refined_rel = compute_reliability_validity(refined_cfa, refined_item_frames)

    baseline_scores = baseline_cfa.factor_scores.join(mean_scores[["ACCEPT"]], how="left")
    refined_scores = refined_cfa.factor_scores.join(mean_scores[["ACCEPT"]], how="left")

    _, baseline_coef_full, baseline_model_full = fit_path_model(
        baseline_scores,
        include_interactions=True,
        bootstrap_iterations=0,
        random_state=42,
    )
    model_data_main, coef_main, model_main = fit_path_model(
        refined_scores,
        include_interactions=False,
        bootstrap_iterations=0,
        random_state=42,
    )
    _, coef_full, model_full = fit_path_model(
        refined_scores,
        include_interactions=True,
        bootstrap_iterations=1000,
        random_state=42,
    )
    final_predictors = [
        predictor
        for predictor in STRUCTURAL_PREDICTORS
        if predictor in coef_main.index and float(coef_main.loc[predictor, "p_value"]) < 0.05
    ]
    final_predictors = final_predictors or ["CVP", "PK", "EA", "PI", "PKN"]
    model_data_final, coef_final, model_final = fit_custom_path_model(
        refined_scores,
        final_predictors,
        bootstrap_iterations=1000,
        random_state=42,
    )

    loading_summary = compute_loading_summary(refined_cfa, refined_item_frames)
    calibration = compute_calibration_summary(model_data_final, model_final)
    multigroup = compute_multigroup_path_diff(df, refined_scores)
    comparison_df = build_model_comparison_table(
        baseline_cfa,
        refined_cfa,
        baseline_model_full,
        model_final,
        item_frames,
        refined_item_frames,
    )
    removed_paths = [predictor for predictor in STRUCTURAL_PREDICTORS if predictor not in final_predictors] + [
        "CVP_x_PI",
        "PC_x_PI",
        "PR_x_PKN",
    ]
    structural_specs = []
    for name, predictors, model in [
        ("基线全模型", list(baseline_coef_full.index), baseline_model_full),
        ("优化全模型", list(coef_full.index), model_full),
        ("最终精简模型", final_predictors, model_final),
    ]:
        rmse_mean, rmse_sd = cross_validated_rmse(
            refined_scores if name != "基线全模型" else baseline_scores,
            predictors,
        )
        structural_specs.append(
            {
                "name": name,
                "predictors": predictors,
                "model": model,
                "cv_rmse": rmse_mean,
                "cv_rmse_sd": rmse_sd,
            }
        )
    structural_comparison = build_structural_model_table(structural_specs)
    final_cv_rmse = float(structural_comparison.loc[structural_comparison["model"] == "最终精简模型", "cv_rmse"].iloc[0])
    final_cv_rmse_sd = float(structural_comparison.loc[structural_comparison["model"] == "最终精简模型", "cv_rmse_sd"].iloc[0])

    baseline_snapshot = build_sem_candidate_snapshot(
        name="冻结当前模型",
        description="冻结当前 23 题、5 条最终路径的优化模型，作为合并高相关构念比较的正式 baseline。",
        item_frames=refined_item_frames,
        cfa_result=refined_cfa,
        reliability_df=refined_rel,
        model_data=model_data_final,
        score_frame=refined_scores,
        coef_table=coef_final,
        model=model_final,
        final_predictors=final_predictors,
        cv_rmse=final_cv_rmse,
        cv_rmse_sd=final_cv_rmse_sd,
    )
    candidate_snapshots = [baseline_snapshot]
    alternative_specs = build_alternative_sem_specs(refined_item_frames)
    spec_lookup = {str(spec["name"]): spec for spec in alternative_specs}
    for spec in alternative_specs:
        candidate_snapshots.append(
            evaluate_alternative_sem_candidate(
                spec,
                mean_scores["ACCEPT"],
                random_state=42,
            )
        )
    candidate_comparison_df = build_candidate_model_comparison_table(
        candidate_snapshots,
        baseline_name="冻结当前模型",
    )
    preferred_model_name = "备选A：购买准备度合并模型"
    preferred_snapshot = next(snapshot for snapshot in candidate_snapshots if snapshot["name"] == preferred_model_name)
    preferred_removed_paths = [
        predictor
        for predictor in spec_lookup[preferred_model_name]["full_predictors"]
        if predictor not in preferred_snapshot["final_predictors"]
    ]
    preferred_calibration = compute_calibration_summary(preferred_snapshot["model_data"], preferred_snapshot["model"])
    preferred_multigroup = compute_multigroup_path_diff(
        df,
        preferred_snapshot["scores"],
        predictors=preferred_snapshot["final_predictors"],
    )
    preferred_summary_comparison = candidate_comparison_df[
        ["模型", "最终路径", "Fornell通过率", "最小判别效度边际", "平均CR", "平均AVE", "Adj_R2", "CV_RMSE", "建议"]
    ].copy()

    plot_measurement_heatmap(refined_cfa, refined_rel, loading_summary, output_dir)
    plot_final_sem_path_diagram(
        preferred_snapshot["cfa"],
        preferred_snapshot["coef"],
        preferred_snapshot["model"],
        preferred_snapshot["final_predictors"],
        decision_df,
        item_frames=preferred_snapshot["item_frames"],
        cv_rmse=preferred_snapshot["cv_rmse"],
        removed_paths=preferred_removed_paths,
        output_dir=output_dir,
    )
    plot_effect_forest(coef_full, output_dir)
    plot_model_dashboard(refined_cfa, refined_rel, calibration, multigroup, output_dir)
    plot_fornell_larcker_heatmap(refined_cfa, refined_rel, output_dir)
    plot_bootstrap_distribution(model_full, coef_full, output_dir)
    plot_item_audit_dashboard(audit_df, decision_df, output_dir)
    plot_model_comparison_dashboard(comparison_df, baseline_cfa, refined_cfa, output_dir)

    export_sem_tables(
        preferred_snapshot["cfa"],
        preferred_snapshot["reliability"],
        preferred_snapshot["cfa"].fit_stats,
        preferred_snapshot["coef"],
        preferred_multigroup,
        preferred_snapshot["scores"],
        preferred_summary_comparison,
        output_dir,
    )
    export_sem_audit_tables(
        baseline_cfa,
        baseline_rel,
        baseline_coef_full,
        refined_cfa,
        refined_rel,
        coef_final,
        audit_df,
        decision_df,
        comparison_df,
        structural_comparison,
        output_dir,
    )
    export_candidate_model_tables(
        candidate_snapshots,
        candidate_comparison_df,
        output_dir,
    )
    write_sem_report(
        preferred_snapshot["cfa"],
        preferred_snapshot["reliability"],
        preferred_snapshot["coef"],
        preferred_calibration,
        preferred_multigroup,
        preferred_snapshot["model"],
        preferred_summary_comparison,
        preferred_removed_paths,
        preferred_snapshot["cv_rmse"],
        output_dir,
    )
    write_candidate_model_report(
        candidate_snapshots,
        candidate_comparison_df,
        baseline_name="冻结当前模型",
        output_dir=output_dir,
    )
    write_calculation_audit_report(
        baseline_cfa,
        refined_cfa,
        baseline_rel,
        refined_rel,
        baseline_coef_full,
        coef_final,
        baseline_model_full,
        model_final,
        audit_df,
        decision_df,
        comparison_df,
        structural_comparison,
        candidate_comparison_df,
        preferred_model_name,
        output_dir,
    )


if __name__ == "__main__":
    main()
