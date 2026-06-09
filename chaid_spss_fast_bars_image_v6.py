"""
chaid_spss_fast.py

Fast SPSS-like CHAID (non-Exhaustive) implementation.

Goal: keep the *tree decisions identical* to a reference "slow but straightforward" implementation,
while reducing runtime by:
- avoiding DataFrame slicing in inner loops
- computing per-predictor sufficient statistics once per node (contingency tables or ANOVA summaries)
- doing merges using table row-sums instead of repeated `isin` filtering

This is engineered for cases like:
- ~3,000 rows, ~200 predictors, modest category counts / interval-binned continuous predictors.

Important notes:
- For categorical targets, results are exact given identical preprocessing and floating behavior of scipy.
- For continuous targets, ANOVA p-values are computed from sufficient statistics (equivalent to scipy f_oneway).
- Case weights are supported for categorical targets and for weighted-ANOVA in the same practical way
  as the companion compat implementation (df2 uses sum(weights)-k).

If you need Graphviz output, install graphviz-python and system graphviz.
"""

from __future__ import annotations

import math
import html
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    import graphviz
except Exception:  # pragma: no cover
    graphviz = None

_MISSING_CAT = "__MISSING__"


def _norm_var_type(v: Optional[str]) -> str:
    if v is None:
        return "nominal"
    v0 = str(v).strip().lower()
    if v0 in {"n", "nom", "nominal", "categorical"}:
        return "nominal"
    if v0 in {"o", "ord", "ordinal"}:
        return "ordinal"
    if v0 in {"s", "scale", "continuous", "cont", "numeric"}:
        return "continuous"
    return "nominal"


@dataclass
class CHAIDNode:
    data: pd.DataFrame
    depth: int
    node_id: int
    parent: Optional["CHAIDNode"] = None
    decision_rule: Optional[str] = None
    is_terminal: bool = False

    split_variable: Optional[str] = None
    split_groups: Optional[List[List[Any]]] = None
    split_p_value: Optional[float] = None
    raw_p_value: Optional[float] = None

    children: List["CHAIDNode"] = None
    prediction: Any = None
    n_eff: float = 0.0

    def __post_init__(self):
        self.children = [] if self.children is None else self.children

    def calculate_prediction(self, target_type: str, target_col: str, weight_col: Optional[str]):
        if target_type == "continuous":
            y = pd.to_numeric(self.data[target_col], errors="coerce")
            if weight_col is None:
                self.prediction = float(y.mean())
                self.n_eff = float(y.notna().sum())
                return
            w = pd.to_numeric(self.data[weight_col], errors="coerce")
            m = y.notna() & w.notna() & (w > 0)
            if m.sum() == 0:
                self.prediction = float(y.mean())
                self.n_eff = float(y.notna().sum())
                return
            self.prediction = float(np.average(y[m].values, weights=w[m].values))
            self.n_eff = float(w[m].sum())
            return

        # categorical target: store probability dict
        y = self.data[target_col].astype(object)
        if weight_col is None:
            y = y.fillna(_MISSING_CAT)
            vc = y.value_counts(dropna=False)
            probs = (vc / vc.sum()).to_dict()
            self.prediction = probs
            self.n_eff = float(vc.sum())
            return

        w = pd.to_numeric(self.data[weight_col], errors="coerce").fillna(0.0)
        y = y.fillna(_MISSING_CAT)
        tmp = pd.DataFrame({"y": y, "w": w})
        agg = tmp.groupby("y", dropna=False)["w"].sum()
        tot = float(agg.sum())
        probs = (agg / tot).to_dict() if tot > 0 else {}
        self.prediction = probs
        self.n_eff = float(tot)


@dataclass
class _PredStats:
    # category axis (original predictor values in deterministic order)
    cats: List[Any]                 # len=k
    # per-row category code in [0,k) or -1 if excluded
    codes: np.ndarray               # shape (n_rows,)
    # effective counts per original category (weighted for categorical target; weighted for cont target too)
    cat_counts: np.ndarray          # shape (k,)
    # target info (categorical only)
    t_cats: Optional[List[Any]] = None
    base_table: Optional[np.ndarray] = None   # (k, m)
    # target info (continuous only)
    sum_y: Optional[np.ndarray] = None        # (k,)
    sumsq_y: Optional[np.ndarray] = None      # (k,)
    n_y: Optional[np.ndarray] = None          # (k,)
    # weights (for weighted ANOVA)
    sum_w: Optional[np.ndarray] = None        # (k,)
    sum_wy: Optional[np.ndarray] = None       # (k,)
    sum_wy2: Optional[np.ndarray] = None      # (k,)


class CHAIDTree:
    def __init__(
        self,
        *,
        method: str = "CHAID",                 # only "CHAID" supported in fast version
        max_depth: int = 3,
        min_parent_size: float = 100.0,
        min_child_size: float = 50.0,
        alpha_merge: float = 0.05,
        alpha_split: float = 0.05,
        n_intervals: int = 10,
        max_intervals: int = 64,
        interval_method: str = "equal_width",  # "equal_width" | "quantile"
        adjust: str = "bonferroni",            # "bonferroni" | "none"
        include_missing_as_category: bool = True,
        allow_resplit: bool = True,
        chi2_statistic: str = "pearson",       # "pearson" | "likelihood_ratio"
    ):
        self.method = str(method).strip().upper()
        if self.method != "CHAID":
            raise ValueError("chaid_spss_fast supports method='CHAID' only (non-Exhaustive).")

        self.max_depth = int(max_depth)
        self.min_parent_size = float(min_parent_size)
        self.min_child_size = float(min_child_size)
        self.alpha_merge = float(alpha_merge)
        self.alpha_split = float(alpha_split)

        self.n_intervals = int(n_intervals)
        self.max_intervals = int(max_intervals)

        self.interval_method = str(interval_method).strip().lower()
        if self.interval_method not in {"equal_width", "quantile"}:
            raise ValueError("interval_method must be 'equal_width' or 'quantile'")

        self.adjust = str(adjust).strip().lower()
        if self.adjust not in {"bonferroni", "none"}:
            raise ValueError("adjust must be 'bonferroni' or 'none'")

        self.include_missing_as_category = bool(include_missing_as_category)
        self.allow_resplit = bool(allow_resplit)

        self.chi2_statistic = str(chi2_statistic).strip().lower()
        if self.chi2_statistic not in {"pearson", "likelihood_ratio"}:
            raise ValueError("chi2_statistic must be 'pearson' or 'likelihood_ratio'")

        self.root: Optional[CHAIDNode] = None
        self.node_count: int = 0

        self.target_col: str = ""
        self.target_type: str = "nominal"
        self.weight_col: Optional[str] = None

        self.variable_types: Dict[str, str] = {}
        self.bin_thresholds: Dict[str, np.ndarray] = {}
        self.processed_predictors: List[str] = []
        self.original_predictor_name: Dict[str, str] = {}

    # ---------- Public API ----------

    def fit(
        self,
        df: pd.DataFrame,
        *,
        target_col: str,
        independent_cols: Sequence[str],
        variable_types: Optional[Dict[str, str]] = None,
        weight_col: Optional[str] = None,
    ) -> "CHAIDTree":
        self.target_col = target_col
        self.weight_col = weight_col

        if self.weight_col is not None and self.weight_col not in df.columns:
            raise ValueError(f"weight_col '{weight_col}' not found in df")

        var_types = dict(variable_types) if variable_types else {}

        target_decl = _norm_var_type(var_types.get(target_col))
        if target_decl == "continuous":
            self.target_type = "continuous"
        else:
            if pd.api.types.is_numeric_dtype(df[target_col]) and df[target_col].nunique(dropna=True) > 10:
                self.target_type = "continuous"
            else:
                self.target_type = "nominal"

        self.df_processed = df.copy()

        # SPSS: missing DV typically excluded; keep as-is for processing and handle masks in stats builder
        if self.target_type != "continuous" and self.include_missing_as_category:
            self.df_processed[target_col] = self.df_processed[target_col].astype(object).fillna(_MISSING_CAT)

        self.variable_types = {}
        self.bin_thresholds = {}
        self.processed_predictors = []
        self.original_predictor_name = {}

        for col in independent_cols:
            if col not in self.df_processed.columns:
                raise ValueError(f"independent column '{col}' not found in df")
            v_type = _norm_var_type(var_types.get(col))

            if v_type == "continuous":
                binned, bins = self._bin_continuous(self.df_processed[col], q=self.n_intervals)
                new_col = f"{col}__binned"
                suffix = 1
                while new_col in self.df_processed.columns:
                    suffix += 1
                    new_col = f"{col}__binned{suffix}"
                self.df_processed[new_col] = binned
                self.variable_types[new_col] = "ordinal"
                self.bin_thresholds[new_col] = bins
                self.processed_predictors.append(new_col)
                self.original_predictor_name[new_col] = col
            else:
                if self.include_missing_as_category:
                    self.df_processed[col] = self.df_processed[col].astype(object).fillna(_MISSING_CAT)
                self.variable_types[col] = "ordinal" if v_type == "ordinal" else "nominal"
                self.processed_predictors.append(col)

        self.root = CHAIDNode(self.df_processed, depth=0, node_id=0)
        self.root.calculate_prediction(self.target_type, self.target_col, self.weight_col)
        self.node_count = 1

        self._build_recursive(self.root)
        return self

    def to_graphviz(
        self,
        *,
        bar_mode: str = "ascii",
        image_dir: str = "chaid_viz_assets",
        bar_width_px: int = 140,
        bar_height_px: int = 14,
        bar_orientation: str = "horizontal",
        sort_children_by: str = "auto",
        positive_class: Optional[Any] = None,
    ):
        """Return a graphviz.Digraph representing the tree.

        bar_mode:
          - "ascii": use ####.... bars (most compatible)
          - "image": generate small PNG bar images and embed via <IMG SRC=...> in node labels

        bar_orientation (only when bar_mode="image"):
          - "horizontal": fill left-to-right (default)
          - "vertical": fill bottom-to-top (SPSS-like mini bar look inside the table)

        sort_children_by (visual order only; does NOT change tree decisions):
          - "auto": continuous/binned splits keep natural interval order; nominal splits order children by increasing positive-class rate (default)
          - "none": keep build order
          - "positive_asc": order children left-to-right by increasing positive-class rate
          - "positive_desc": order children left-to-right by decreasing positive-class rate

        positive_class:
          - Which class is considered "positive" for sorting.
            If None, we auto-detect common binaries (yes/no, true/false, 1/0) per node.

        image_dir is used only when bar_mode="image". It must be reachable from the directory where
        you call dot.render(...). The simplest is to keep image_dir relative to your notebook working dir
        and render into the same directory.
        """
        if graphviz is None:
            raise RuntimeError("graphviz is not installed/importable in this environment.")

        self._viz_bar_mode = str(bar_mode).strip().lower()
        if self._viz_bar_mode not in {"ascii", "image"}:
            raise ValueError('bar_mode must be "ascii" or "image"')

        self._viz_bar_orientation = str(bar_orientation).strip().lower()
        if self._viz_bar_orientation not in {"horizontal", "vertical"}:
            raise ValueError('bar_orientation must be "horizontal" or "vertical"')

        # If you switched to vertical mini-bars but left the default (horizontal) size,
        # auto-adjust to a more SPSS-like aspect ratio.
        if self._viz_bar_mode == "image" and self._viz_bar_orientation == "vertical":
            if int(bar_width_px) == 140 and int(bar_height_px) == 14:
                bar_width_px, bar_height_px = 14, 60

        # Visual ordering only (does not change split decisions)
        sc = str(sort_children_by).strip().lower()
        if sc in {"pos_asc", "positive_rate_asc", "rate_asc"}:
            sc = "positive_asc"
        if sc in {"pos_desc", "positive_rate_desc", "rate_desc"}:
            sc = "positive_desc"
        if sc not in {"auto", "none", "positive_asc", "positive_desc"}:
            raise ValueError('sort_children_by must be "auto", "none", "positive_asc", or "positive_desc"')
        self._viz_sort_children_by = sc
        self._viz_positive_class = positive_class

        self._viz_image_dir = str(image_dir)
        self._viz_bar_size = (int(bar_width_px), int(bar_height_px))

        if self._viz_bar_mode == "image":
            self._ensure_viz_image_dir()

        dot = graphviz.Digraph(comment="CHAID Tree", format="png")
        dot.attr(rankdir="TB", splines="polyline")
        dot.attr("node", shape="plaintext", fontname="Arial", fontsize="10")
        self._add_graphviz_node(dot, self.root)
        return dot
    def _eff_n(self, data: pd.DataFrame) -> float:
        if self.weight_col is None:
            if self.target_type == "continuous":
                return float(pd.to_numeric(data[self.target_col], errors="coerce").notna().sum())
            return float(len(data))
        w = pd.to_numeric(data[self.weight_col], errors="coerce").fillna(0.0)
        if self.target_type == "continuous":
            y = pd.to_numeric(data[self.target_col], errors="coerce")
            m = y.notna() & (w > 0)
            return float(w[m].sum())
        return float(w[w > 0].sum())

    def _ordered_unique_categories(self, s: pd.Series, pred_type: str) -> List[Any]:
        vals = list(pd.unique(s))
        # keep deterministic order similar to compat: ordinal sorted, nominal str-sort
        if pred_type == "ordinal":
            # numeric ordinals sort numerically, else by str
            if pd.api.types.is_numeric_dtype(s):
                missing_val = -1
                non_missing = [v for v in vals if not (pd.isna(v) or v == missing_val)]
                non_missing_sorted = sorted(non_missing)
                if missing_val in vals or any(pd.isna(v) for v in vals):
                    return non_missing_sorted + [missing_val]
                return non_missing_sorted
            missing_val = _MISSING_CAT
            non_missing = [v for v in vals if not (pd.isna(v) or v == missing_val)]
            non_missing_sorted = sorted(non_missing, key=lambda x: str(x))
            if missing_val in vals or any(pd.isna(v) for v in vals):
                return non_missing_sorted + [missing_val]
            return non_missing_sorted

        # nominal
        if not self.include_missing_as_category:
            vals = [v for v in vals if not pd.isna(v)]
        else:
            vals = [(_MISSING_CAT if pd.isna(v) else v) for v in vals]
        return sorted(vals, key=lambda x: str(x))

    def _pair_adjust(self, p: float, n_comp: int) -> float:
        if self.adjust != "bonferroni":
            return float(min(p, 1.0))
        return float(min(p * max(1, int(n_comp)), 1.0))

    def _apply_split_adjustment(self, p_val: float, *, c: int, r: int, pred_type: str) -> float:
        p_val = float(p_val)
        if p_val >= 1.0 or self.adjust != "bonferroni":
            return min(p_val, 1.0)
        c = int(c)
        r = int(r)
        if c <= 1 or r <= 1:
            return 1.0
        if pred_type == "ordinal":
            try:
                B = math.comb(c - 1, r - 1)
            except Exception:
                B = 1
        else:
            sum_val = 0.0
            for i in range(r):
                sum_val += ((-1.0) ** i) * ((r - i) ** c) / (math.factorial(i) * math.factorial(r - i))
            B = max(1.0, sum_val)
        return float(min(p_val * B, 1.0))

    def _chi2_p(self, observed: np.ndarray) -> float:
        """P-value for chi-square (Pearson or likelihood ratio) computed like compat.

        compat uses 1 - CDF, which can underflow to 0 for extremely small p-values.
        We intentionally keep that behavior to preserve split tie-breaking.
        """
        if observed.ndim != 2 or observed.shape[0] < 2 or observed.shape[1] < 2:
            return 1.0
        obs = np.asarray(observed, dtype=float)
        if np.any(obs < 0):
            return 1.0
        row_sum = obs.sum(axis=1, keepdims=True)
        col_sum = obs.sum(axis=0, keepdims=True)
        total = float(obs.sum())
        if total <= 0:
            return 1.0
        exp = row_sum @ col_sum / total
        df = int((obs.shape[0] - 1) * (obs.shape[1] - 1))
        if df <= 0:
            return 1.0
        if self.chi2_statistic == "pearson":
            with np.errstate(divide="ignore", invalid="ignore"):
                chi2 = float(np.nansum((obs - exp) ** 2 / exp))
            p = float(1.0 - stats.chi2.cdf(chi2, df))
            return 1.0 if (np.isnan(p) or p < 0) else p
        # likelihood ratio (G^2)
        with np.errstate(divide="ignore", invalid="ignore"):
            mask = (obs > 0) & (exp > 0)
            g2 = float(2.0 * np.nansum(obs[mask] * np.log(obs[mask] / exp[mask])))
        p = float(1.0 - stats.chi2.cdf(g2, df))
        return 1.0 if (np.isnan(p) or p < 0) else p

    def _anova_p_from_summaries(
        self,
        *,
        n: np.ndarray,
        sum_y: np.ndarray,
        sumsq_y: np.ndarray,
    ) -> float:
        # unweighted one-way ANOVA from sufficient statistics (equivalent to scipy f_oneway)
        n = n.astype(float)
        if n.sum() < 2:
            return 1.0
        k = int(np.sum(n > 0))
        if k < 2:
            return 1.0
        # filter empty groups
        m = n > 0
        n = n[m]
        sum_y = sum_y[m].astype(float)
        sumsq_y = sumsq_y[m].astype(float)

        N = float(n.sum())
        overall_mean = float(sum_y.sum() / N)
        means = sum_y / n
        ssb = float(np.sum(n * (means - overall_mean) ** 2))
        ssw = float(np.sum(sumsq_y - n * (means ** 2)))
        df1 = k - 1
        df2 = int(max(1.0, N - k))
        msb = ssb / df1 if df1 > 0 else 0.0
        msw = ssw / df2 if df2 > 0 else 0.0
        if msw <= 0:
            return 1.0
        F = msb / msw
        return float(stats.f.sf(F, df1, df2))

    def _weighted_anova_p_from_summaries(
        self,
        *,
        sum_w: np.ndarray,
        sum_wy: np.ndarray,
        sum_wy2: np.ndarray,
    ) -> float:
        # matches compat's weighted approach: df2 uses n_eff=sum(weights)-k
        m = sum_w > 0
        if m.sum() < 2:
            return 1.0
        sum_w = sum_w[m].astype(float)
        sum_wy = sum_wy[m].astype(float)
        sum_wy2 = sum_wy2[m].astype(float)

        k = int(len(sum_w))
        n_eff = float(sum_w.sum())
        if k < 2 or n_eff <= 0:
            return 1.0
        overall_mean = float(sum_wy.sum() / n_eff)

        means = sum_wy / sum_w
        ssb = float(np.sum(sum_w * (means - overall_mean) ** 2))
        # within: sum(w*(y-mean)^2) = sum(w*y^2) - 2*mean*sum(w*y) + mean^2*sum(w)
        ssw = float(np.sum(sum_wy2 - 2.0 * means * sum_wy + (means ** 2) * sum_w))

        df1 = k - 1
        df2 = max(1.0, n_eff - k)
        msb = ssb / df1 if df1 > 0 else 0.0
        msw = ssw / df2 if df2 > 0 else 0.0
        if msw <= 0:
            return 1.0
        F = msb / df1 / (ssw / df2) if ssw > 0 else 0.0
        # keep exact with our msb/msw:
        F = msb / msw
        return float(1.0 - stats.f.cdf(F, df1, df2))

    def _bin_continuous(self, s: pd.Series, *, q: int) -> Tuple[pd.Series, np.ndarray]:
        s_num = pd.to_numeric(s, errors="coerce")
        non_missing = s_num.dropna()
        if non_missing.empty:
            return pd.Series([-1] * len(s_num), index=s.index), np.array([np.nan, np.nan])

        q_eff = int(max(2, min(q, self.max_intervals, non_missing.nunique())))
        if self.interval_method == "quantile":
            try:
                binned, bins = pd.qcut(s_num, q=q_eff, labels=False, duplicates="drop", retbins=True)
            except Exception:
                uniq = np.unique(non_missing.values)
                if len(uniq) <= 1:
                    binned = pd.Series([0] * len(s_num), index=s.index)
                    bins = np.array([uniq[0], uniq[0]])
                else:
                    q_eff = min(q_eff, len(uniq))
                    binned, bins = pd.qcut(s_num, q=q_eff, labels=False, duplicates="drop", retbins=True)
        else:
            # equal-width
            mn = float(non_missing.min())
            mx = float(non_missing.max())
            if mn == mx:
                binned = pd.Series([0] * len(s_num), index=s.index)
                bins = np.array([mn, mx])
            else:
                bins = np.linspace(mn, mx, num=q_eff + 1)
                binned = pd.cut(s_num, bins=bins, labels=False, include_lowest=True, right=True)
        binned = binned.astype("float")
        if self.include_missing_as_category:
            binned = binned.fillna(-1)
        return binned.astype(int), np.asarray(bins)

    # ---------- Stats building per predictor per node ----------

    def _build_pred_stats(self, data: pd.DataFrame, predictor: str, pred_type: str) -> Optional[_PredStats]:
        # Exclude rows with missing DV (SPSS-like)
        if self.target_type == "continuous":
            y = pd.to_numeric(data[self.target_col], errors="coerce")
            if self.weight_col is None:
                dv_mask = y.notna()
            else:
                w = pd.to_numeric(data[self.weight_col], errors="coerce")
                dv_mask = y.notna() & w.notna() & (w > 0)
        else:
            y = data[self.target_col].astype(object)
            if self.include_missing_as_category:
                y = y.fillna(_MISSING_CAT)
            dv_mask = y.notna()  # after fillna always True, but keep shape

            if self.weight_col is not None:
                w = pd.to_numeric(data[self.weight_col], errors="coerce").fillna(0.0)
                dv_mask = dv_mask & (w > 0)

        x = data[predictor].astype(object)
        if self.include_missing_as_category:
            x = x.fillna(_MISSING_CAT)

        x = x[dv_mask]
        if len(x) < 2:
            return None

        cats = self._ordered_unique_categories(x, pred_type.lower())
        if len(cats) < 2:
            return None

        cat_to_idx = {c: i for i, c in enumerate(cats)}
        codes = x.map(cat_to_idx).to_numpy()
        # codes is valid by construction
        k = len(cats)

        if self.target_type == "continuous":
            yv = pd.to_numeric(data.loc[dv_mask, self.target_col], errors="coerce").to_numpy()
            if self.weight_col is None:
                # unweighted summaries
                n_y = np.bincount(codes, minlength=k).astype(float)
                sum_y = np.bincount(codes, weights=yv, minlength=k).astype(float)
                sumsq_y = np.bincount(codes, weights=yv * yv, minlength=k).astype(float)
                return _PredStats(
                    cats=cats,
                    codes=codes,
                    cat_counts=n_y.copy(),
                    sum_y=sum_y,
                    sumsq_y=sumsq_y,
                    n_y=n_y,
                )
            # weighted summaries
            wv = pd.to_numeric(data.loc[dv_mask, self.weight_col], errors="coerce").to_numpy()
            sum_w = np.bincount(codes, weights=wv, minlength=k).astype(float)
            sum_wy = np.bincount(codes, weights=wv * yv, minlength=k).astype(float)
            sum_wy2 = np.bincount(codes, weights=wv * yv * yv, minlength=k).astype(float)
            return _PredStats(
                cats=cats,
                codes=codes,
                cat_counts=sum_w.copy(),
                sum_w=sum_w,
                sum_wy=sum_wy,
                sum_wy2=sum_wy2,
            )

        # categorical target stats
        yv = y.loc[dv_mask].astype(object)
        # factorize target (order doesn't matter for chi2)
        t_codes, t_uniques = pd.factorize(yv, sort=False)
        m = len(t_uniques)
        if m < 2:
            return None

        if self.weight_col is None:
            weights = None
        else:
            weights = pd.to_numeric(data.loc[dv_mask, self.weight_col], errors="coerce").fillna(0.0).to_numpy()

        # build base contingency: k x m
        idx = codes.astype(int) * m + t_codes.astype(int)
        if weights is None:
            flat = np.bincount(idx, minlength=k * m).astype(float)
        else:
            flat = np.bincount(idx, weights=weights, minlength=k * m).astype(float)
        base = flat.reshape(k, m)
        cat_counts = base.sum(axis=1)
        return _PredStats(
            cats=cats,
            codes=codes,
            cat_counts=cat_counts,
            t_cats=list(t_uniques),
            base_table=base,
        )

    # ---------- Grouping evaluation using stats ----------

    def _groups_to_table(self, ps: _PredStats, groups_idx: List[List[int]]) -> np.ndarray:
        # sum rows for each group
        rows = []
        for g in groups_idx:
            rows.append(ps.base_table[np.array(g, dtype=int), :].sum(axis=0))
        return np.vstack(rows)

    def _p_for_groups(self, ps: _PredStats, groups_idx: List[List[int]]) -> float:
        if len(groups_idx) < 2:
            return 1.0
        if self.target_type == "continuous":
            if self.weight_col is None:
                n = np.array([ps.n_y[np.array(g, dtype=int)].sum() for g in groups_idx], dtype=float)
                sy = np.array([ps.sum_y[np.array(g, dtype=int)].sum() for g in groups_idx], dtype=float)
                sy2 = np.array([ps.sumsq_y[np.array(g, dtype=int)].sum() for g in groups_idx], dtype=float)
                return self._anova_p_from_summaries(n=n, sum_y=sy, sumsq_y=sy2)
            sw = np.array([ps.sum_w[np.array(g, dtype=int)].sum() for g in groups_idx], dtype=float)
            swy = np.array([ps.sum_wy[np.array(g, dtype=int)].sum() for g in groups_idx], dtype=float)
            swy2 = np.array([ps.sum_wy2[np.array(g, dtype=int)].sum() for g in groups_idx], dtype=float)
            return self._weighted_anova_p_from_summaries(sum_w=sw, sum_wy=swy, sum_wy2=swy2)

        observed = self._groups_to_table(ps, groups_idx)
        return self._chi2_p(observed)

    def _best_pair_merge_p(self, ps: _PredStats, groups_idx: List[List[int]], pred_type: str) -> Tuple[Optional[Tuple[int, int]], float, int]:
        # return pair indices, raw p, n_comp
        if pred_type == "nominal":
            pairs = list(combinations(range(len(groups_idx)), 2))
        else:
            pairs = [(i, i + 1) for i in range(len(groups_idx) - 1)]
        if not pairs:
            return None, 1.0, 1

        best_p = -1.0
        best_pair = None
        # deterministic tie-break: first encountered
        for i, j in pairs:
            p = self._p_for_groups(ps, [groups_idx[i], groups_idx[j]])
            if p > best_p:
                best_p = p
                best_pair = (i, j)
        return best_pair, float(best_p), max(1, len(pairs))

    def _merge_small_segments_idx(self, ps: _PredStats, pred_type: str, groups_idx: List[List[int]]) -> List[List[int]]:
        # Merge groups whose effective size < min_child_size with most similar other group.
        # Use pairwise p as similarity measure (same as compat).
        if len(groups_idx) < 2:
            return groups_idx

        while True:
            sizes = [float(ps.cat_counts[np.array(g, dtype=int)].sum()) for g in groups_idx]
            small = [i for i, sz in enumerate(sizes) if sz < self.min_child_size]
            if not small or len(groups_idx) < 2:
                break

            i = small[0]
            # choose best partner (highest p); nominal: all, ordinal: neighbors
            candidates: List[int] = []
            if pred_type == "nominal":
                candidates = [j for j in range(len(groups_idx)) if j != i]
            else:
                if i - 1 >= 0:
                    candidates.append(i - 1)
                if i + 1 < len(groups_idx):
                    candidates.append(i + 1)
            if not candidates:
                break

            best_p = -1.0
            best_j = None
            for j in candidates:
                p = self._p_for_groups(ps, [groups_idx[i], groups_idx[j]])
                if p > best_p:
                    best_p = p
                    best_j = j

            if best_j is None:
                break

            a, b = (i, best_j) if i < best_j else (best_j, i)
            merged = groups_idx[a] + groups_idx[b]
            new_groups = []
            for k in range(len(groups_idx)):
                if k == a:
                    new_groups.append(merged)
                elif k == b:
                    continue
                else:
                    new_groups.append(groups_idx[k])
            groups_idx = new_groups

        return groups_idx

    def _handle_missing_ordinal_idx(self, ps: _PredStats, groups_idx: List[List[int]]) -> List[List[int]]:
        # For ordinal predictors, missing category often encoded as last cat (either -1 or __MISSING__).
        # SPSS can merge missing with the most similar category, or keep separate if it is distinct.
        # Here: if a singleton missing group exists, merge it with the neighbor that yields the largest p
        # if that p (adjusted for 2 comps) is > alpha_merge; else keep.
        # This is a pragmatic SPSS-like behavior and matches the compat intent.
        if len(groups_idx) < 2:
            return groups_idx
        missing_idx = None
        # identify missing category index by value
        for ci, v in enumerate(ps.cats):
            if v == _MISSING_CAT or v == -1:
                missing_idx = ci
                break
        if missing_idx is None:
            return groups_idx
        # find group that contains missing only
        gpos = None
        for gi, g in enumerate(groups_idx):
            if len(g) == 1 and g[0] == missing_idx:
                gpos = gi
                break
        if gpos is None:
            return groups_idx

        # candidates: neighbors
        candidates = []
        if gpos - 1 >= 0:
            candidates.append(gpos - 1)
        if gpos + 1 < len(groups_idx):
            candidates.append(gpos + 1)
        if not candidates:
            return groups_idx

        best_p = -1.0
        best_j = None
        for j in candidates:
            p = self._p_for_groups(ps, [groups_idx[gpos], groups_idx[j]])
            if p > best_p:
                best_p = p
                best_j = j
        if best_j is None:
            return groups_idx

        adj_p = self._pair_adjust(best_p, n_comp=len(candidates))
        if adj_p > self.alpha_merge:
            a, b = (gpos, best_j) if gpos < best_j else (best_j, gpos)
            merged = groups_idx[a] + groups_idx[b]
            new_groups = []
            for k in range(len(groups_idx)):
                if k == a:
                    new_groups.append(merged)
                elif k == b:
                    continue
                else:
                    new_groups.append(groups_idx[k])
            return new_groups

        return groups_idx

    def _chaid_groups_fast(self, ps: _PredStats, pred_type: str) -> Tuple[List[List[int]], float, float]:
        # groups are indices into ps.cats
        pred_type = pred_type.lower()
        groups = [[i] for i in range(len(ps.cats))]
        original_c = len(groups)

        history: List[Tuple[List[List[int]], float, float]] = []

        while True:
            # SPSS-like post-processing for evaluation
            g_eval = [g.copy() for g in groups]
            g_eval = self._merge_small_segments_idx(ps, pred_type, g_eval)
            if pred_type == "ordinal":
                g_eval = self._handle_missing_ordinal_idx(ps, g_eval)

            raw_p = self._p_for_groups(ps, g_eval)
            adj_p = self._apply_split_adjustment(raw_p, c=original_c, r=len(g_eval), pred_type=pred_type)
            history.append(([g.copy() for g in g_eval], float(raw_p), float(adj_p)))

            if len(groups) <= 1 or len(groups) == 2:
                break

            best_pair, best_pair_p, n_comp = self._best_pair_merge_p(ps, groups, pred_type)
            if best_pair is None:
                break

            adj_merge_p = self._pair_adjust(best_pair_p, n_comp)
            if adj_merge_p > self.alpha_merge:
                i, j = best_pair
                merged = groups[i] + groups[j]
                if pred_type == "ordinal":
                    groups[i] = merged
                    del groups[j]
                else:
                    a, b = (i, j) if i < j else (j, i)
                    new_groups = []
                    for k in range(len(groups)):
                        if k == a:
                            new_groups.append(merged)
                        elif k == b:
                            continue
                        else:
                            new_groups.append(groups[k])
                    groups = new_groups
                continue

            break

        if self.allow_resplit and history:
            best_groups, best_raw, best_adj = min(history, key=lambda t: t[2])
            return best_groups, float(best_raw), float(best_adj)

        last_groups, last_raw, last_adj = history[-1]
        return last_groups, float(last_raw), float(last_adj)

    # ---------- Tree build ----------

    def _build_recursive(self, node: CHAIDNode):
        node.calculate_prediction(self.target_type, self.target_col, self.weight_col)

        if node.depth >= self.max_depth:
            node.is_terminal = True
            return
        if self._eff_n(node.data) < self.min_parent_size:
            node.is_terminal = True
            return
        if self.target_type != "continuous":
            if node.data[self.target_col].nunique(dropna=False) <= 1:
                node.is_terminal = True
                return

        best_split = None
        best_adj_p = 1.0

        for predictor in self.processed_predictors:
            if node.data[predictor].nunique(dropna=False) <= 1:
                continue
            pred_type = self.variable_types.get(predictor, "nominal")
            ps = self._build_pred_stats(node.data, predictor, pred_type)
            if ps is None:
                continue

            groups_idx, raw_p, adj_p = self._chaid_groups_fast(ps, pred_type)
            if not groups_idx or len(groups_idx) < 2:
                continue

            # child size check
            ok = True
            for g in groups_idx:
                sz = float(ps.cat_counts[np.array(g, dtype=int)].sum())
                if sz < self.min_child_size:
                    ok = False
                    break
            if not ok:
                continue

            if adj_p < best_adj_p - 1e-12:
                best_adj_p = float(adj_p)
                # convert group indices -> original values
                groups_values = [[ps.cats[i] for i in g] for g in groups_idx]
                best_split = {
                    "predictor": predictor,
                    "groups": groups_values,
                    "raw_p": float(raw_p),
                    "adj_p": float(adj_p),
                }

        if best_split is None or best_split["adj_p"] >= self.alpha_split:
            node.is_terminal = True
            return

        node.split_variable = best_split["predictor"]
        node.split_groups = best_split["groups"]
        node.split_p_value = best_split["adj_p"]
        node.raw_p_value = best_split["raw_p"]

        for group in node.split_groups:
            mask = node.data[node.split_variable].isin(group)
            child_data = node.data.loc[mask].copy()
            rule_desc = self._generate_label(node.split_variable, group)

            child = CHAIDNode(
                child_data,
                depth=node.depth + 1,
                node_id=self.node_count,
                parent=node,
                decision_rule=rule_desc,
            )
            self.node_count += 1
            # store split group for visualization ordering (no effect on decisions)
            try:
                child._split_group = group
            except Exception:
                pass
            node.children.append(child)
            self._build_recursive(child)

    # ---------- Labels + viz ----------

    def _generate_label(self, variable: str, group: List[Any]) -> str:
        if variable in self.bin_thresholds:
            bins = self.bin_thresholds[variable]
            # missing bucket
            if any((isinstance(v, (int, np.integer)) and v < 0) or (v == _MISSING_CAT) for v in group):
                non_missing = [v for v in group if not (isinstance(v, (int, np.integer)) and v < 0) and v != _MISSING_CAT]
                parts = ["Missing"]
                if non_missing:
                    parts.append(self._generate_label(variable, non_missing))
                return " or ".join(parts)

            idxs = sorted(int(v) for v in group)
            min_idx = idxs[0]
            max_idx = idxs[-1]
            min_idx = max(0, min(min_idx, len(bins) - 2))
            max_idx = max(0, min(max_idx, len(bins) - 2))

            lower = bins[min_idx]
            upper = bins[max_idx + 1]

            is_min = (min_idx == 0)
            is_max = (max_idx == len(bins) - 2)

            if is_min and is_max:
                return "all"
            if is_min:
                return f"<= {upper:.6g}"
            if is_max:
                return f"> {lower:.6g}"
            return f"({lower:.6g}, {upper:.6g}]"

        if len(group) > 12:
            return f"[{group[0]} ... {group[-1]}]"
        return str(group).replace("'", "")

    def _text_bar(self, p: float, width: int = 18, full: str = "#", empty: str = ".") -> str:
        """Return a fixed-width ASCII bar for probability p in [0,1].
    
        We keep ASCII by default for maximum compatibility in Graphviz HTML labels.
        """
        try:
            p = float(p)
        except Exception:
            p = 0.0
        if not np.isfinite(p):
            p = 0.0
        p = max(0.0, min(1.0, p))
        filled = int(round(p * width))
        filled = max(0, min(width, filled))
        return (full * filled) + (empty * (width - filled))

    def _label_token(self, x: Any) -> str:
        """Normalize a class label for stable yes/no, true/false, 1/0 ordering."""
        try:
            import numpy as _np
        except Exception:
            _np = None

        # Booleans
        if isinstance(x, (bool,)) or (_np is not None and isinstance(x, (_np.bool_,))):
            return "true" if bool(x) else "false"

        # 0/1 numbers (int/float)
        if isinstance(x, (int,)) or (_np is not None and isinstance(x, (_np.integer,))):
            if int(x) == 1:
                return "1"
            if int(x) == 0:
                return "0"

        if isinstance(x, (float,)) or (_np is not None and isinstance(x, (_np.floating,))):
            if float(x) == 1.0:
                return "1"
            if float(x) == 0.0:
                return "0"

        return str(x).strip().lower()

    def _order_binary_items(self, items: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
        """If items has exactly 2 classes, prefer SPSS-like ordering for common binaries."""
        if len(items) != 2:
            return items

        norm = {self._label_token(k): (k, v) for k, v in items}

        # Common binary pairs
        pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("1", "0"),
            ("y", "n"),
            ("t", "f"),
        ]
        for a, b in pairs:
            if a in norm and b in norm and len(norm) == 2:
                return [norm[a], norm[b]]

        # Fallback: sort by label string
        return sorted(items, key=lambda kv: str(kv[0]))


    def _detect_positive_key(self, keys: Sequence[Any], *, preferred: Optional[Any] = None) -> Optional[Any]:
        """Detect which class should be treated as 'positive' for display sorting.

        - If preferred is given, we try to match it to an existing key (by token normalization first, then exact).
        - Otherwise, we auto-detect common binaries: yes/no, true/false, 1/0.
        Returns the actual key object from `keys`, or None if not detected.
        """
        keys_list = list(keys)
        if not keys_list:
            return None

        token_to_key = {self._label_token(k): k for k in keys_list}

        def match_pref(pref: Any) -> Optional[Any]:
            if pref is None:
                return None
            # try token match
            t = self._label_token(pref)
            if t in token_to_key:
                return token_to_key[t]
            # try exact match
            for k in keys_list:
                if k == pref:
                    return k
            # try string-equal match (common when keys are ints/bools and preferred is str)
            ps = str(pref)
            for k in keys_list:
                if str(k) == ps:
                    return k
            return None

        pk = match_pref(preferred)
        if pk is not None:
            return pk

        # auto-detect binaries
        pairs = [
            ("yes", "no"),
            ("true", "false"),
            ("1", "0"),
            ("y", "n"),
            ("t", "f"),
        ]
        for pos, neg in pairs:
            if pos in token_to_key and neg in token_to_key and len(token_to_key) == 2:
                return token_to_key[pos]

        return None

    def _positive_prob_for_node(self, node: CHAIDNode, *, positive_key: Optional[Any]) -> Optional[float]:
        """Return probability of positive_key for a node (categorical target)."""
        if positive_key is None:
            return None
        if not isinstance(node.prediction, dict):
            return None
        try:
            v = node.prediction.get(positive_key, None)
            return float(v) if v is not None else 0.0
        except Exception:
            return 0.0

    def _ensure_viz_image_dir(self) -> None:
        """Ensure directory for per-node bar images exists (bar_mode='image')."""
        import os
        os.makedirs(getattr(self, "_viz_image_dir", "chaid_viz_assets"), exist_ok=True)
        if not hasattr(self, "_viz_bar_cache"):
            self._viz_bar_cache = {}

    def _render_prob_bar_image(self, *, prob: float, node_id: int, row_idx: int) -> str:
        """Render a tiny bar PNG representing prob in [0,1]. Returns a posix path.

        Orientation is controlled by self._viz_bar_orientation ("horizontal" or "vertical").
        """
        import os
        from PIL import Image, ImageDraw

        self._ensure_viz_image_dir()
        w, h = getattr(self, "_viz_bar_size", (140, 14))
        orientation = getattr(self, "_viz_bar_orientation", "horizontal")
        try:
            p = float(prob)
        except Exception:
            p = 0.0
        if not np.isfinite(p):
            p = 0.0
        p = max(0.0, min(1.0, p))

        # Cache by rounded prob + size to avoid regenerating.
        key = (int(node_id), int(row_idx), int(w), int(h), str(orientation), round(p, 6))
        cached = self._viz_bar_cache.get(key)
        if cached and os.path.exists(cached):
            return cached.replace("\\", "/")

        # include prob in filename to avoid collisions when re-rendering with same node/row
        prob_tag = int(round(p * 10000))
        o_tag = "h" if str(orientation).lower().startswith("h") else "v"
        filename = f"node{int(node_id)}_row{int(row_idx)}_{o_tag}_{prob_tag:04d}.png"
        path = os.path.join(self._viz_image_dir, filename)

        # Create image (white background + gray border + filled bar)
        img = Image.new("RGBA", (int(w), int(h)), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        border = (160, 160, 160, 255)
        bg = (255, 255, 255, 255)
        fill = (70, 130, 180, 255)  # steelblue-ish

        draw.rectangle([0, 0, int(w) - 1, int(h) - 1], fill=bg, outline=border)

        inner_w = max(1, int(w) - 2)
        inner_h = max(1, int(h) - 2)

        if str(orientation).lower().startswith("v"):
            fill_h = int(round(p * inner_h))
            if fill_h > 0:
                y0 = 1 + (inner_h - fill_h)
                draw.rectangle([1, y0, 1 + inner_w, 1 + inner_h], fill=fill)
        else:
            fill_w = int(round(p * inner_w))
            if fill_w > 0:
                draw.rectangle([1, 1, 1 + fill_w, 1 + inner_h], fill=fill)

        img.save(path)

        self._viz_bar_cache[key] = path
        return path.replace("\\", "/")


    def _render_distribution_chart_image(self, *, probs: List[float], node_id: int, tag: str = "dist") -> str:
        """Render a tiny multi-bar chart PNG for a node distribution.

        - Bars are laid out horizontally (side-by-side) for easier comparison.
        - Fill direction inside each bar is controlled by self._viz_bar_orientation:
            * "vertical": bottom-to-top fill (classic column chart)
            * "horizontal": left-to-right fill inside each bar box
        Returns a POSIX-style path suitable for Graphviz <IMG SRC="...">.
        """
        import os
        from PIL import Image, ImageDraw

        self._ensure_viz_image_dir()
        w_box, h_box = getattr(self, "_viz_bar_size", (14, 60))
        orientation = getattr(self, "_viz_bar_orientation", "vertical")

        # sanitize probs
        pp: List[float] = []
        for p in probs:
            try:
                p = float(p)
            except Exception:
                p = 0.0
            if not np.isfinite(p):
                p = 0.0
            pp.append(max(0.0, min(1.0, p)))

        n = len(pp)
        if n <= 0:
            # empty placeholder
            filename = f"node{int(node_id)}_{tag}_empty.png"
            path = os.path.join(self._viz_image_dir, filename)
            if not os.path.exists(path):
                img = Image.new("RGBA", (max(4, int(w_box)), max(4, int(h_box))), (255, 255, 255, 0))
                img.save(path)
            return path.replace("\\", "/")

        gap = max(2, int(w_box) // 3)
        W = 2 + (n * int(w_box)) + ((n - 1) * gap)
        H = 2 + int(h_box)

        # Cache by distribution vector + size/orientation
        key = ("chart", int(node_id), int(w_box), int(h_box), int(gap), str(orientation), tuple(round(x, 6) for x in pp))
        cached = self._viz_bar_cache.get(key)
        if cached and os.path.exists(cached):
            return cached.replace("\\", "/")

        # include a stable hash so reruns don't overwrite unexpectedly
        h = abs(hash(key)) % (10**10)
        o_tag = "v" if str(orientation).lower().startswith("v") else "h"
        filename = f"node{int(node_id)}_{tag}_{o_tag}_{h}.png"
        path = os.path.join(self._viz_image_dir, filename)

        img = Image.new("RGBA", (int(W), int(H)), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        border = (160, 160, 160, 255)
        bg = (255, 255, 255, 255)
        fill = (70, 130, 180, 255)

        # Draw each bar box
        for i, p in enumerate(pp):
            x0 = 1 + i * (int(w_box) + gap)
            y0 = 1
            x1 = x0 + int(w_box) - 1
            y1 = y0 + int(h_box) - 1

            draw.rectangle([x0, y0, x1, y1], fill=bg, outline=border)

            inner_w = max(1, int(w_box) - 2)
            inner_h = max(1, int(h_box) - 2)

            if str(orientation).lower().startswith("v"):
                fill_h = int(round(p * inner_h))
                if fill_h > 0:
                    fy0 = y0 + 1 + (inner_h - fill_h)
                    draw.rectangle([x0 + 1, fy0, x0 + 1 + inner_w, y0 + 1 + inner_h], fill=fill)
            else:
                fill_w = int(round(p * inner_w))
                if fill_w > 0:
                    draw.rectangle([x0 + 1, y0 + 1, x0 + 1 + fill_w, y0 + 1 + inner_h], fill=fill)

        img.save(path)
        self._viz_bar_cache[key] = path
        return path.replace("\\", "/")

    def _add_graphviz_node(self, dot, node: Optional[CHAIDNode]):
        if node is None:
            return
    
        bg_color = "#FFFFE0"
        viz_mode = getattr(self, "_viz_bar_mode", "ascii")
        # For image-mode categorical targets, embed one grouped chart row and keep the table compact.
        ncols = 3 if (viz_mode == "image" and self.target_type != "continuous") else 4
    
        rows: List[str] = []
        rows.append(
            f'<TR><TD COLSPAN="{ncols}" BGCOLOR="{bg_color}" BORDER="1"><B>Node {node.node_id}</B></TD></TR>'
        )
    
        if self.target_type == "continuous":
            rows.append(
                f'<TR><TD COLSPAN="{ncols}" BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT">'
                f'Mean: {node.prediction:.4g}<BR/>n_eff: {node.n_eff:.4g}</TD></TR>'
            )
        else:
            # Compute class counts at render-time so we can show both % and (weighted) counts like SPSS.
            y = node.data[self.target_col].astype(object)
            if self.include_missing_as_category:
                y = y.fillna(_MISSING_CAT)

            if self.weight_col is None:
                counts = y.value_counts(dropna=False)
                total = float(counts.sum())
            else:
                w = pd.to_numeric(node.data[self.weight_col], errors="coerce").fillna(0.0)
                tmp = pd.DataFrame({"y": y, "w": w})
                counts = tmp.groupby("y", dropna=False)["w"].sum().sort_values(ascending=False)
                total = float(counts.sum())

            if total <= 0:
                total = float(node.n_eff) if node.n_eff else 0.0

            items = list(counts.items())

            # Prefer SPSS-like ordering for common binaries when exactly 2 classes.
            if len(items) == 2:
                items = self._order_binary_items(items)

            if getattr(self, "_viz_bar_mode", "ascii") == "image":
                # ---- Grouped chart: vertical mini-bars laid out horizontally (side-by-side) ----
                # Show all if binary; otherwise show top-k (and bucket the rest as "Other").
                if len(items) <= 2:
                    shown = items
                    extra = []
                else:
                    max_show = 8
                    shown = items[:max_show]
                    extra = items[max_show:]

                if extra:
                    other_cnt = float(sum(float(c) for _, c in extra))
                    shown = list(shown) + [("Other", other_cnt)]

                probs = [(float(cnt) / total) if total > 0 else 0.0 for _, cnt in shown]
                chart_path = self._render_distribution_chart_image(probs=probs, node_id=node.node_id, tag="dist")
                # Prevent Graphviz from stretching the chart differently between notebook preview and file render.
                # We embed the image at its native pixel size and disable scaling.
                w_box, h_box = getattr(self, "_viz_bar_size", (14, 60))
                gap = max(2, int(w_box) // 3)
                n_bars = len(probs)
                chart_w = 2 + (n_bars * int(w_box)) + ((n_bars - 1) * gap) if n_bars > 0 else int(w_box)
                chart_h = 2 + int(h_box)
                chart_cell = f'<IMG SRC="{html.escape(chart_path)}" SCALE="FALSE"/>'

                # Header row
                rows.append(
                    f'<TR>'
                    f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT"><B>Class</B></TD>'
                    f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="RIGHT"><B>%</B></TD>'
                    f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="RIGHT"><B>Count</B></TD>'
                    f'</TR>'
                )

                # Chart row (spans all columns)
                rows.append(
                    f'<TR><TD COLSPAN="{ncols}" BGCOLOR="{bg_color}" BORDER="1" ALIGN="CENTER" '                    f'FIXEDSIZE="TRUE" WIDTH="{int(chart_w)}" HEIGHT="{int(chart_h)}">'
                    f'{chart_cell}</TD></TR>'
                )

                # Rows for classes (same order as chart bars)
                for cls, cnt in shown:
                    cls_txt = html.escape(str(cls))
                    cnt_f = float(cnt)
                    prob = (cnt_f / total) if total > 0 else 0.0
                    rows.append(
                        f'<TR>'
                        f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT">{cls_txt}</TD>'
                        f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="RIGHT">{prob:.1%}</TD>'
                        f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="RIGHT">{cnt_f:.4g}</TD>'
                        f'</TR>'
                    )

                if extra:
                    rows.append(
                        f'<TR><TD COLSPAN="{ncols}" BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT">'
                        f'+{len(extra)} more classes</TD></TR>'
                    )

            else:
                # ---- Row-wise ASCII bars (most compatible) ----
                rows.append(
                    f'<TR>'
                    f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT"><B>Class</B></TD>'
                    f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT"><B>Distribution</B></TD>'
                    f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="RIGHT"><B>%</B></TD>'
                    f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="RIGHT"><B>Count</B></TD>'
                    f'</TR>'
                )

                max_show = len(items) if len(items) <= 2 else 8
                for row_i, (cls, cnt) in enumerate(items[:max_show]):
                    cls_txt = html.escape(str(cls))
                    cnt_f = float(cnt)
                    prob = (cnt_f / total) if total > 0 else 0.0
                    bar = self._text_bar(prob, width=18, full="#", empty=".")
                    dist_cell = f'<FONT FACE="Courier">{bar}</FONT>'
                    rows.append(
                        f'<TR>'
                        f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT">{cls_txt}</TD>'
                        f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT">{dist_cell}</TD>'
                        f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="RIGHT">{prob:.1%}</TD>'
                        f'<TD BGCOLOR="{bg_color}" BORDER="1" ALIGN="RIGHT">{cnt_f:.4g}</TD>'
                        f'</TR>'
                    )

                if len(items) > max_show:
                    rows.append(
                        f'<TR><TD COLSPAN="{ncols}" BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT">'
                        f'+{len(items) - max_show} more classes</TD></TR>'
         )
    
            rows.append(
                f'<TR><TD COLSPAN="{ncols}" BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT">'
                f'n_eff: {node.n_eff:.4g}</TD></TR>'
            )
    
        if node.split_variable:
            disp = self.original_predictor_name.get(node.split_variable, node.split_variable)
            disp = html.escape(str(disp))
            rows.append(
                f'<TR><TD COLSPAN="{ncols}" BGCOLOR="{bg_color}" BORDER="1" ALIGN="LEFT">'
                f'<B>Split:</B> {disp}<BR/>Adj p: {node.split_p_value:.4g} (raw {node.raw_p_value:.4g})</TD></TR>'
            )
    
        label = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">' + "".join(rows) + "</TABLE>>"
        dot.node(str(node.node_id), label)
    
        # Child ordering for display (visual only; does NOT change tree decisions)
        children = list(node.children)
        if node.split_variable and children:
            sc = getattr(self, "_viz_sort_children_by", "auto")

            # 1) Continuous/binned splits: keep natural interval order (ascending).
            if node.split_variable in self.bin_thresholds:
                def _interval_key(ch: CHAIDNode):
                    g = getattr(ch, "_split_group", None)
                    if not g:
                        return (float("inf"), ch.node_id)
                    non = []
                    for v in g:
                        if isinstance(v, (int, np.integer)) and int(v) >= 0:
                            non.append(int(v))
                    if non:
                        return (min(non), ch.node_id)
                    return (float("inf"), ch.node_id)

                children.sort(key=_interval_key)
            else:
                # 2) Nominal (unordered) splits: order by increasing positive-class rate by default.
                pred_type = self.variable_types.get(node.split_variable, "nominal")
                if sc == "auto" and pred_type == "nominal" and self.target_type != "continuous":
                    sc = "positive_asc"

                if sc in {"positive_asc", "positive_desc"} and self.target_type != "continuous":
                    # Determine positive class key
                    pos_pref = getattr(self, "_viz_positive_class", None)
                    pos_key = None
                    if isinstance(node.prediction, dict):
                        pos_key = self._detect_positive_key(node.prediction.keys(), preferred=pos_pref)
                    if pos_key is None and children and isinstance(children[0].prediction, dict):
                        pos_key = self._detect_positive_key(children[0].prediction.keys(), preferred=pos_pref)

                    if pos_key is not None:
                        scored: List[Tuple[float, int, CHAIDNode]] = []
                        for ch in children:
                            p = self._positive_prob_for_node(ch, positive_key=pos_key)
                            p = float(p) if p is not None else 0.0
                            scored.append((p, ch.node_id, ch))
                        scored.sort(key=lambda t: (t[0], t[1]), reverse=(sc == "positive_desc"))
                        children = [t[2] for t in scored]

        for child in children:
            self._add_graphviz_node(dot, child)
            edge_label = child.decision_rule if child.decision_rule else ""
            dot.edge(str(node.node_id), str(child.node_id), label=edge_label)


# -------------------------------
# Notebook-friendly helpers (optional)
# -------------------------------

def infer_variable_types(
    df: "pd.DataFrame",
    *,
    target_col: str,
    independent_cols: "Sequence[str]",
    treat_binary_numeric_as_nominal: bool = True,
    overrides: "Optional[Dict[str, str]]" = None,
) -> "Dict[str, str]":
    """Infer variable types for CHAID.

    Rules (simple, editable):
      - target defaults to nominal unless override says continuous.
      - predictors: numeric -> continuous, non-numeric -> nominal.
      - if treat_binary_numeric_as_nominal=True, numeric predictors with nunique<=2 are set to nominal.
      - apply overrides last.

    Types: 'nominal' | 'continuous' (also accepts 'ordinal' but treated as nominal internally).
    """
    vt: Dict[str, str] = {target_col: "nominal"}
    for c in independent_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            nun = df[c].nunique(dropna=True)
            if treat_binary_numeric_as_nominal and nun <= 2:
                vt[c] = "nominal"
            else:
                vt[c] = "continuous"
        else:
            vt[c] = "nominal"
    if overrides:
        for k, v in overrides.items():
            vt[k] = str(v).strip().lower()
    return vt


def collapse_rare_categories(
    df: "pd.DataFrame",
    col: str,
    *,
    min_count: int = 30,
    other_label: str = "(other)",
    include_na: bool = True,
) -> "pd.Series":
    """Collapse rare categories into a single '(other)' label (useful to cap branch count).

    This changes the *input data* (so tree can change), but is often preferable to forcing
    a max-branches rule because it is explicit and reproducible.
    """
    s = df[col].copy()
    if include_na:
        s = s.fillna("(missing)")
    vc = s.value_counts(dropna=False)
    rare = vc[vc < min_count].index
    s = s.where(~s.isin(rare), other_label)
    return s


def drop_high_cardinality_columns(
    df: "pd.DataFrame",
    cols: "Sequence[str]",
    *,
    unique_ratio_threshold: float = 0.9,
    max_unique: int = 500,
) -> "List[str]":
    """Suggest columns to drop (IDs etc.) based on high cardinality."""
    drop: List[str] = []
    n = len(df)
    for c in cols:
        nun = df[c].nunique(dropna=True)
        if nun >= max_unique or (n > 0 and (nun / n) >= unique_ratio_threshold):
            drop.append(c)
    return drop
