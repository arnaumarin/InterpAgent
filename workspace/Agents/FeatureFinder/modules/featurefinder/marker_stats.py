"""
Marker-feature statistics for FeatureFinder.

- Every test runs on ONE ROW PER PROMPT. Tokens of the same prompt are not independent
  observations, so token-level (`all_tokens`) data is averaged per prompt before testing.
- Wilcoxon scores/p-values, log fold changes and Cohen's d are all computed on the raw
  (non-negative) SAE codes. scanpy's own `logfoldchanges` assumes log1p input and applies
  expm1 to the group means, which is wrong for raw codes, so it is replaced here.
- `split_half_validation` selects markers on half of each category's prompts and tests
  them on the other half, so a category can actually fail validation.

Only numpy/pandas/scipy/scanpy are needed (no torch), so this is unit-testable on its own.
"""

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse, stats

DEFAULT_THRESHOLDS = {
    "max_pval_adj": 0.001,   # BH-adjusted Wilcoxon p-value
    "min_score": 2.0,        # Wilcoxon z-score
    "min_logfc": 0.5,        # log2(mean in category / mean in rest), raw SAE codes
    "min_effect_size": 0.3,  # Cohen's d, raw SAE codes
}
LOGFC_PSEUDOCOUNT = 1e-9


def _dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def benjamini_hochberg(pvals):
    p = np.nan_to_num(np.asarray(pvals, dtype=float), nan=1.0)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def group_stats(X, in_group):
    """Per-column mean in group, mean in rest, log2 fold change and Cohen's d."""
    X = _dense(X).astype(np.float64)
    a, b = X[in_group], X[~in_group]
    mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
    logfc = np.log2((mean_a + LOGFC_PSEUDOCOUNT) / (mean_b + LOGFC_PSEUDOCOUNT))
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return mean_a, mean_b, logfc, np.zeros(X.shape[1])
    pooled = np.sqrt(((na - 1) * a.var(axis=0, ddof=1) + (nb - 1) * b.var(axis=0, ddof=1)) / (na + nb - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(pooled > 0, (mean_a - mean_b) / pooled, 0.0)
    return mean_a, mean_b, logfc, d


def prompt_level(adata):
    """AnnData with one row per prompt: mean raw SAE code over the prompt's tokens."""
    X = adata.raw.X if adata.raw is not None else adata.X
    var_names = adata.raw.var_names if adata.raw is not None else adata.var_names
    obs = adata.obs
    if "prompt_id" not in obs:
        raise ValueError("adata.obs needs a 'prompt_id' column")
    codes, prompt_ids = pd.factorize(obs["prompt_id"])
    categories = obs.groupby(codes, sort=True)["category"].agg(lambda c: c.unique().tolist())
    if (categories.str.len() > 1).any():
        raise ValueError("Some prompt_id values span more than one category")
    if len(prompt_ids) == obs.shape[0]:  # already one row per prompt (factorize keeps row order)
        Xp = _dense(X)
    else:
        n = obs.shape[0]
        G = sparse.csr_matrix((np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(len(prompt_ids), n))
        Xp = _dense(G @ X) / np.bincount(codes)[:, None]
    pobs = pd.DataFrame({
        "prompt_id": np.asarray(prompt_ids, dtype=str),
        "category": pd.Categorical(categories.str[0].values),
        "n_tokens": np.bincount(codes),
    })
    pobs.index = pobs["prompt_id"].values
    return ad.AnnData(X=np.asarray(Xp, dtype=np.float32), obs=pobs, var=pd.DataFrame(index=var_names))


def rank_markers(pdata, n_genes=100, groups="all"):
    """Wilcoxon (category vs rest, BH) on prompt-level raw codes, with correct log fold changes."""
    sc.tl.rank_genes_groups(pdata, groupby="category", groups=groups, method="wilcoxon", use_raw=False,
                            n_genes=n_genes, corr_method="benjamini-hochberg")
    rgg = pdata.uns["rank_genes_groups"]
    lfc = rgg["logfoldchanges"].copy()
    categories = pdata.obs["category"].astype(str).values
    for group in rgg["names"].dtype.names:
        cols = pdata.var_names.get_indexer(rgg["names"][group])
        lfc[group] = group_stats(pdata.X[:, cols], categories == group)[2]
    rgg["logfoldchanges"] = lfc
    return pdata


def select_markers(pdata, category, thresholds=None):
    """Markers for `category` that pass every threshold, sorted by Wilcoxon score."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    df = sc.get.rank_genes_groups_df(pdata, group=category, key="rank_genes_groups")
    cols = pdata.var_names.get_indexer(df["names"])
    in_group = (pdata.obs["category"].astype(str) == category).values
    mean_in, mean_rest, logfc, d = group_stats(pdata.X[:, cols], in_group)
    df["logfoldchanges"] = logfc
    df["effect_size"] = d
    df["mean_in_category"] = mean_in
    df["mean_rest"] = mean_rest
    keep = ((df["pvals_adj"] < t["max_pval_adj"]) & (df["scores"] > t["min_score"])
            & (df["logfoldchanges"] > t["min_logfc"]) & (df["effect_size"] > t["min_effect_size"]))
    return df[keep].sort_values("scores", ascending=False).reset_index(drop=True)


def split_half_validation(pdata, thresholds=None, n_genes=100, seed=0, min_markers=1,
                          min_replication=0.5, alpha=0.05, min_prompts_per_half=3):
    """Select markers on a random half of each category's prompts; test them on the other half.

    A marker replicates if, on the held-out half, its Cohen's d exceeds the selection
    threshold and a one-sided Mann-Whitney test (category > rest, BH within the category)
    gives p_adj < alpha. A category is valid if at least `min_markers` markers were found
    on the discovery half and at least `min_replication` of them replicate.
    """
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rng = np.random.default_rng(seed)
    categories = pdata.obs["category"].astype(str).values
    discovery = np.zeros(pdata.n_obs, dtype=bool)
    for category in np.unique(categories):
        idx = rng.permutation(np.where(categories == category)[0])
        discovery[idx[: len(idx) // 2]] = True

    n_disc = {c: int(((categories == c) & discovery).sum()) for c in np.unique(categories)}
    n_held = {c: int(((categories == c) & ~discovery).sum()) for c in np.unique(categories)}
    eligible = [c for c in n_disc if min(n_disc[c], n_held[c]) >= min_prompts_per_half]
    disc = rank_markers(pdata[discovery].copy(), n_genes=n_genes, groups=eligible) if eligible else None
    held = pdata[~discovery]
    held_categories = categories[~discovery]
    counts = pd.Series(categories).value_counts()
    balance_ratio = float(counts.min() / counts.max())

    results = {}
    for category in sorted(n_disc):
        row = {"n_discovery_prompts": n_disc[category], "n_heldout_prompts": n_held[category],
               "balance_ratio": balance_ratio, "seed": seed}
        if category not in eligible:
            results[category] = {**row, "valid": False, "n_discovery_markers": 0,
                                 "reason": f"Too few prompts to split ({n_disc[category]}/{n_held[category]} per half)"}
            continue
        markers = select_markers(disc, category, t)
        if len(markers) < min_markers:
            results[category] = {**row, "valid": False, "n_discovery_markers": len(markers),
                                 "reason": f"{len(markers)} markers on the discovery half (< {min_markers})"}
            continue
        cols = held.var_names.get_indexer(markers["names"])
        Xh = _dense(held.X[:, cols]).astype(np.float64)
        in_group = held_categories == category
        d_held = group_stats(Xh, in_group)[3]
        p_held = stats.mannwhitneyu(Xh[in_group], Xh[~in_group], alternative="greater", axis=0).pvalue
        replicated = (d_held > t["min_effect_size"]) & (benjamini_hochberg(p_held) < alpha)
        rate = float(replicated.mean())
        valid = rate >= min_replication
        results[category] = {
            **row,
            "valid": bool(valid),
            "reason": "" if valid else f"Only {rate:.0%} of discovery markers replicate (< {min_replication:.0%})",
            "n_discovery_markers": len(markers),
            "n_replicated": int(replicated.sum()),
            "replication_rate": rate,
            "median_heldout_effect_size": float(np.median(d_held)),
            "replicated_markers": markers["names"][replicated].tolist(),
        }
    return results


def permutation_test(pdata, n_permutations=100, alpha=0.05, n_genes=100, seed=0):
    """Count of BH-significant markers per category vs. counts under shuffled prompt labels."""
    work = pdata.copy()
    original_labels = work.obs["category"].astype(str).values
    categories = np.unique(original_labels)

    def n_significant():
        rank_markers(work, n_genes=n_genes)
        out = {}
        for category in categories:
            df = sc.get.rank_genes_groups_df(work, group=category, key="rank_genes_groups")
            out[category] = int((df["pvals_adj"] < 0.05).sum())
        return out

    original = n_significant()
    rng = np.random.default_rng(seed)
    permuted = {c: [] for c in categories}
    for _ in range(n_permutations):
        work.obs["category"] = pd.Categorical(rng.permutation(original_labels))
        for c, n in n_significant().items():
            permuted[c].append(n)

    pvalues = {c: float((1 + sum(n >= original[c] for n in permuted[c])) / (1 + n_permutations))
               for c in categories}
    n_sig = sum(p < alpha for p in pvalues.values())
    return {
        "permutation_pvalues": pvalues,
        "original_counts": original,
        "n_significant_categories": int(n_sig),
        "overall_significant": bool(n_sig > 0),
        "n_permutations": n_permutations,
        "unit": "prompt",
    }
