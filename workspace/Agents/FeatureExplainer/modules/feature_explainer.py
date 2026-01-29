"""
Feature explanation module adapted for FeatureExplainer.
Generates hypotheses and explanations for SAE features discovered by FeatureFinder.
Includes full hypothesis/test/refine loop with SAE interface.
Seven-metric evaluation and ranking (Detection, Fuzzing, Surprisal, Embedding, P-value, Cohen's d, LLM Judge).
"""

import os
import re
import openai
import pandas as pd
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import Counter

try:
    from scipy.stats import ttest_ind, rankdata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False


@dataclass
class FeatureHypothesis:
    """Hypothesis about what an SAE feature detects."""
    feature_id: str
    description: str
    confidence: float
    reasoning: str
    language_specificity: Optional[str] = None
    semantic_category: Optional[str] = None
    iteration: int = 0
    refinement_history: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FeatureExplanationReport:
    """Comprehensive explanation report for a feature."""
    feature_id: str
    feature_index: int
    layer: int
    one_sentence_explanation: str
    detailed_explanation: str
    top_activating_categories: List[str]
    confidence_score: float
    hypothesis_evolution: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Seven-metric evaluation and ranking (paper-style)
# ---------------------------------------------------------------------------

@dataclass
class ComprehensiveEvaluation:
    """Evaluation with the 7 ranking metrics: Detection, Fuzzing, Surprisal, Embedding, P-value, Cohen's d, LLM Judge."""
    hypothesis_description: str
    detection_score: float
    fuzzing_score: float
    surprisal_score: float
    embedding_score: float
    p_value: float
    effect_size: float
    llm_judge_score: float
    detection_rank: int = 0
    fuzzing_rank: int = 0
    surprisal_rank: int = 0
    embedding_rank: int = 0
    p_value_rank: int = 0
    effect_size_rank: int = 0
    llm_judge_rank: int = 0
    average_rank: float = 0.0
    dominance_score: int = 0
    rank_score: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class HypothesisEvaluator:
    """
    Evaluates hypotheses using the 7 metrics.
    Uses fallback LLM-based detection/fuzzing/surprisal (no Delphi dependency).
    Optional: sentence-transformers for embedding; scipy for p-value/effect size.
    """

    def __init__(self, openai_client, model: str = "gpt-4o"):
        self.openai_client = openai_client
        self.model = model
        self._embed_model = None

    def _call_llm(self, prompt: str, temperature: float = 0.0) -> str:
        """Call LLM and return response text."""
        try:
            resp = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=500,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""

    def _fallback_detection(
        self, hypothesis: str, activating: List[str], non_activating: List[str]
    ) -> float:
        """Fallback detection: LLM predicts which examples match the hypothesis; F1."""
        n_each = min(5, len(activating), len(non_activating)) if non_activating else min(5, len(activating))
        if n_each == 0:
            return 0.5
        act_sample = list(np.random.choice(activating, min(n_each, len(activating)), replace=False))
        non_sample = list(np.random.choice(non_activating, min(n_each, len(non_activating)), replace=False)) if non_activating else []
        all_examples = act_sample + non_sample
        labels = [1] * len(act_sample) + [0] * len(non_sample)
        combined = list(zip(all_examples, labels))
        np.random.shuffle(combined)
        examples, true_labels = zip(*combined)
        prompt = f"""Feature: "{hypothesis}"

Which examples match? Answer YES/NO for each:

{chr(10).join([f'{i+1}. {ex[:150]}' for i, ex in enumerate(examples)])}

Format: "1: YES, 2: NO, ...":"""
        response = self._call_llm(prompt, temperature=0.0)
        predicted = []
        for i in range(len(examples)):
            match = re.search(rf"{i+1}[:\s]*(YES|NO)", response, re.IGNORECASE)
            predicted.append(1 if match and "YES" in (match.group(1) or "").upper() else 0)
        tp = sum(1 for i in range(len(examples)) if predicted[i] == 1 and true_labels[i] == 1)
        fp = sum(1 for i in range(len(examples)) if predicted[i] == 1 and true_labels[i] == 0)
        fn = sum(1 for i in range(len(examples)) if predicted[i] == 0 and true_labels[i] == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return float(f1)

    def _fallback_fuzzing(
        self, hypothesis: str, activating: List[str], non_activating: List[str]
    ) -> float:
        """Fallback fuzzing: LLM judges whether highlighted tokens are the activating ones."""
        scores = []
        for example in activating[:5]:
            words = example.replace("\u2581", " ").split()
            if len(words) < 3:
                continue
            n_highlight = min(2, len(words) // 2)
            highlight_idx = np.random.choice(len(words), n_highlight, replace=False)
            highlighted = [f"<<{words[i]}>>" if i in highlight_idx else words[i] for i in range(len(words))]
            highlighted_text = " ".join(highlighted)
            prompt = f"""Feature: "{hypothesis}"
Text: {highlighted_text}

Are <<delimited>> tokens the activating ones? YES/NO:"""
            response = self._call_llm(prompt, temperature=0.0).upper()
            scores.append(1.0 if "YES" in response else 0.0)
        return float(np.mean(scores)) if scores else 0.5

    def _fallback_surprisal(
        self, hypothesis: str, activating: List[str], non_activating: List[str]
    ) -> float:
        """Fallback surprisal: coherence rating for activating vs non-activating; discrimination in [0,1]."""
        act_scores = []
        for ex in activating[:5]:
            prompt = f"""Context: "{hypothesis}"
Text: {ex[:200]}

Coherence rating (0-10):"""
            response = self._call_llm(prompt, temperature=0.0)
            m = re.search(r"\d+(?:\.\d+)?", response)
            act_scores.append(min(float(m.group()) / 10.0, 1.0) if m else 0.5)
        non_act_scores = []
        for ex in (non_activating or [])[:5]:
            prompt = f"""Context: "{hypothesis}"
Text: {ex[:200]}

Coherence rating (0-10):"""
            response = self._call_llm(prompt, temperature=0.0)
            m = re.search(r"\d+(?:\.\d+)?", response)
            non_act_scores.append(min(float(m.group()) / 10.0, 1.0) if m else 0.5)
        avg_act = np.mean(act_scores) if act_scores else 0.5
        avg_non = np.mean(non_act_scores) if non_act_scores else 0.5
        return float(max(0.0, min(1.0, (avg_act - avg_non + 1.0) / 2.0)))

    def _embedding_score(
        self, hypothesis: str, activating: List[str], non_activating: List[str]
    ) -> float:
        """Embedding similarity between hypothesis and examples; discrimination score in [0,1]."""
        if not EMBEDDING_AVAILABLE:
            return 0.5
        try:
            if self._embed_model is None:
                self._embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            n_act = min(5, len(activating))
            n_non = min(5, len(non_activating)) if non_activating else 0
            act_sample = activating[:n_act]
            non_sample = (non_activating or [])[:n_non]
            hyp_emb = self._embed_model.encode(hypothesis, convert_to_tensor=False)
            act_embs = self._embed_model.encode(act_sample, convert_to_tensor=False)
            sim_act = np.dot(act_embs, hyp_emb) / (np.linalg.norm(act_embs, axis=1) * np.linalg.norm(hyp_emb) + 1e-9)
            avg_sim_act = float(np.mean(sim_act))
            if non_sample:
                non_embs = self._embed_model.encode(non_sample, convert_to_tensor=False)
                sim_non = np.dot(non_embs, hyp_emb) / (np.linalg.norm(non_embs, axis=1) * np.linalg.norm(hyp_emb) + 1e-9)
                avg_sim_non = float(np.mean(sim_non))
                return max(0.0, min(1.0, (avg_sim_act - avg_sim_non + 1.0) / 2.0))
            return max(0.0, min(1.0, (avg_sim_act + 1.0) / 2.0))
        except Exception:
            return 0.5

    def _compute_p_value(self, activating: np.ndarray, non_activating: np.ndarray) -> float:
        """Independent-samples t-test p-value (activating > non_activating)."""
        if not SCIPY_AVAILABLE or len(activating) < 2 or len(non_activating) < 2:
            return 0.5
        _, p_value = ttest_ind(activating, non_activating, equal_var=False, alternative="greater")
        return float(p_value)

    def _compute_effect_size(self, activating: np.ndarray, non_activating: np.ndarray) -> float:
        """Cohen's d."""
        if len(activating) < 2 or len(non_activating) < 2:
            return 0.0
        mean_act = np.mean(activating)
        mean_non = np.mean(non_activating)
        std_act = np.std(activating, ddof=1)
        std_non = np.std(non_activating, ddof=1)
        n_act, n_non = len(activating), len(non_activating)
        pooled_std = np.sqrt(((n_act - 1) * std_act**2 + (n_non - 1) * std_non**2) / (n_act + n_non - 2))
        return float((mean_act - mean_non) / pooled_std) if pooled_std > 0 else 0.0

    def _llm_judge_score(self, hypothesis: str, activating_examples: List[str]) -> float:
        """LLM-as-judge: rate how well hypothesis captures the pattern (0-1)."""
        examples_text = "\n".join([f"{i+1}. {ex[:200]}" for i, ex in enumerate(activating_examples[:10])])
        prompt = f"""Rate hypothesis quality (0-10):

HYPOTHESIS: "{hypothesis}"

EXAMPLES:
{examples_text}

How well does this hypothesis capture the pattern (0-10)?

Rating:"""
        response = self._call_llm(prompt, temperature=0.0)
        m = re.search(r"(\d+(?:\.\d+)?)", response)
        if m:
            return min(float(m.group(1)) / 10.0, 1.0)
        return 0.5

    def evaluate(
        self,
        hypothesis: str,
        activating_examples: List[str],
        non_activating_examples: List[str],
        activating_scores: np.ndarray,
        non_activating_scores: np.ndarray,
        n_samples: int = 10,
        verbose: bool = False,
    ) -> ComprehensiveEvaluation:
        """Evaluate one hypothesis with the 7 metrics."""
        if verbose:
            print(f"  Evaluating: '{hypothesis[:60]}...'")
        detection = self._fallback_detection(hypothesis, activating_examples, non_activating_examples)
        fuzzing = self._fallback_fuzzing(hypothesis, activating_examples, non_activating_examples or [])
        surprisal = self._fallback_surprisal(hypothesis, activating_examples, non_activating_examples or [])
        embedding = self._embedding_score(hypothesis, activating_examples, non_activating_examples or [])
        p_value = self._compute_p_value(activating_scores, non_activating_scores)
        effect_size = self._compute_effect_size(activating_scores, non_activating_scores)
        llm_judge = self._llm_judge_score(hypothesis, activating_examples[:n_samples])
        return ComprehensiveEvaluation(
            hypothesis_description=hypothesis,
            detection_score=detection,
            fuzzing_score=fuzzing,
            surprisal_score=surprisal,
            embedding_score=embedding,
            p_value=p_value,
            effect_size=effect_size,
            llm_judge_score=llm_judge,
        )


# Default metric specs for aggregate_rank (column names and direction)
METRIC_SPECS_RANK = {
    "DetF1": "high",
    "FuzzF1": "high",
    "SurpAUROC": "high",
    "Embed": "high",
    "Judge": "high",
    "p_val": "low",
    "d": "high",
}


def aggregate_rank(
    df: pd.DataFrame,
    metric_specs: Dict[str, str],
    id_cols: Tuple[str, ...] = ("Source", "Hypothesis"),
    ties_method: str = "average",
) -> pd.DataFrame:
    """
    Multi-metric rank aggregation.

    Parameters
    ----------
    df : pd.DataFrame
        One row per hypothesis/candidate. Must contain columns for each key in metric_specs.
    metric_specs : dict
        Mapping metric_name -> "high" or "low" for direction.
        Example: {"DetF1": "high", "FuzzF1": "high", "SurpAUROC": "high", "Embed": "high", "Judge": "high", "p_val": "low", "d": "high"}
    id_cols : tuple[str]
        Columns to keep for identification in the output.
    ties_method : str
        Pandas ranking tie method. Use "average" to match paper.

    Returns
    -------
    out : pd.DataFrame
        Includes per-metric rank columns r_<metric>, AvgRank, and final Rank.
    """
    out = df.copy()

    def _to_numeric(series: pd.Series) -> pd.Series:
        s = series.copy()
        if s.dtype == object or s.dtype.name == "string":
            s = s.astype(str).str.strip()
            s = s.replace(r"^<\s*0\.001$", "0.001", regex=True)
        return pd.to_numeric(s, errors="coerce")

    rank_cols = []
    for m, direction in metric_specs.items():
        if m not in out.columns:
            raise ValueError(f"Metric '{m}' not found in DataFrame. Columns: {list(out.columns)}")
        vals = _to_numeric(out[m])
        if vals.isna().any():
            missing = out.loc[vals.isna()]
            raise ValueError(
                f"Metric '{m}' has non-numeric / missing values in rows:\n{missing[list(id_cols) + [m]].to_string(index=False)}"
            )
        ascending = direction.lower() == "low"
        r = vals.rank(ascending=ascending, method=ties_method)
        col = f"r_{m}"
        out[col] = r
        rank_cols.append(col)

    out["AvgRank"] = out[rank_cols].mean(axis=1)
    out["Rank"] = out["AvgRank"].rank(ascending=True, method=ties_method).astype(float)

    cols_front = ["Rank", *id_cols, "AvgRank"]
    cols_rest = [c for c in out.columns if c not in cols_front]
    out = out.sort_values(["AvgRank", *id_cols]).reset_index(drop=True)
    out = out[cols_front + cols_rest]
    return out


def results_to_ranking_df(
    results: List[Tuple[str, ComprehensiveEvaluation]],
) -> pd.DataFrame:
    """Build a DataFrame from (source, ComprehensiveEvaluation) with columns Source, Hypothesis, DetF1, FuzzF1, SurpAUROC, Embed, Judge, p_val, d."""
    rows = []
    for source, e in results:
        p_val = "<0.001" if e.p_value < 0.001 else round(e.p_value, 6)
        rows.append({
            "Source": source,
            "Hypothesis": (e.hypothesis_description or "")[:500],
            "DetF1": round(e.detection_score, 4),
            "FuzzF1": round(e.fuzzing_score, 4),
            "SurpAUROC": round(e.surprisal_score, 4),
            "Embed": round(e.embedding_score, 4),
            "Judge": round(e.llm_judge_score, 4),
            "p_val": p_val,
            "d": round(e.effect_size, 4),
        })
    return pd.DataFrame(rows)


def rank_hypotheses_with_aggregate(
    results: List[Tuple[str, ComprehensiveEvaluation]],
    metric_specs: Optional[Dict[str, str]] = None,
    id_cols: Tuple[str, ...] = ("Source", "Hypothesis"),
    ties_method: str = "average",
) -> pd.DataFrame:
    """Build ranking DataFrame from evaluation results and apply aggregate_rank. Returns ranked DataFrame with Rank, Source, Hypothesis, AvgRank, r_*."""
    if not results:
        return pd.DataFrame()
    df = results_to_ranking_df(results)
    specs = metric_specs or METRIC_SPECS_RANK
    return aggregate_rank(df, specs, id_cols=id_cols, ties_method=ties_method)


def compute_rankings(
    results: List[Tuple[str, ComprehensiveEvaluation]],
) -> List[Tuple[str, ComprehensiveEvaluation]]:
    """Compute rankings for the 7 metrics (Borda-style); sort by dominance then average rank."""
    if not results:
        return results
    n = len(results)
    evals = [r[1] for r in results]
    if not SCIPY_AVAILABLE:
        for i, (_, e) in enumerate(results):
            e.average_rank = 1.0
            e.dominance_score = 0
            e.rank_score = 1.0
        return results
    metrics = {
        "detection": [e.detection_score for e in evals],
        "fuzzing": [e.fuzzing_score for e in evals],
        "surprisal": [e.surprisal_score for e in evals],
        "embedding": [e.embedding_score for e in evals],
        "p_value": [e.p_value for e in evals],
        "effect_size": [e.effect_size for e in evals],
        "llm_judge": [e.llm_judge_score for e in evals],
    }
    rankings: Dict[str, List[float]] = {}
    for metric_name, scores in metrics.items():
        scores_array = np.array(scores, dtype=float)
        nan_mask = np.isnan(scores_array)
        if metric_name == "p_value":
            scores_clean = np.where(nan_mask, np.inf, scores_array)
            ranks = rankdata(scores_clean, method="average")
        else:
            scores_clean = np.where(nan_mask, -np.inf, scores_array)
            ranks = rankdata(-scores_clean, method="average")
        rankings[metric_name] = ranks.tolist()
    for i, (_, eval_obj) in enumerate(results):
        def safe_int_rank(rank_val: float) -> int:
            if np.isnan(rank_val) or np.isinf(rank_val):
                return n
            return int(rank_val)

        eval_obj.detection_rank = safe_int_rank(rankings["detection"][i])
        eval_obj.fuzzing_rank = safe_int_rank(rankings["fuzzing"][i])
        eval_obj.surprisal_rank = safe_int_rank(rankings["surprisal"][i])
        eval_obj.embedding_rank = safe_int_rank(rankings["embedding"][i])
        eval_obj.p_value_rank = safe_int_rank(rankings["p_value"][i])
        eval_obj.effect_size_rank = safe_int_rank(rankings["effect_size"][i])
        eval_obj.llm_judge_rank = safe_int_rank(rankings["llm_judge"][i])
        all_ranks = [
            eval_obj.detection_rank, eval_obj.fuzzing_rank, eval_obj.surprisal_rank,
            eval_obj.embedding_rank, eval_obj.p_value_rank, eval_obj.effect_size_rank, eval_obj.llm_judge_rank,
        ]
        eval_obj.average_rank = float(np.mean(all_ranks))
        dominance_count = 0
        for j in range(n):
            if i == j:
                continue
            other_ranks = [
                evals[j].detection_rank, evals[j].fuzzing_rank, evals[j].surprisal_rank,
                evals[j].embedding_rank, evals[j].p_value_rank, evals[j].effect_size_rank, evals[j].llm_judge_rank,
            ]
            all_better_or_equal = all(all_ranks[k] <= other_ranks[k] for k in range(7))
            at_least_one_better = any(all_ranks[k] < other_ranks[k] for k in range(7))
            if all_better_or_equal and at_least_one_better:
                dominance_count += 1
        eval_obj.dominance_score = dominance_count
        eval_obj.rank_score = 1.0 - (eval_obj.average_rank - 1) / (n - 1) if n > 1 else 1.0
    return sorted(results, key=lambda x: (-x[1].dominance_score, x[1].average_rank))


def print_ranking_analysis(
    results: List[Tuple[str, ComprehensiveEvaluation]],
    metric_specs: Optional[Dict[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """Rank hypotheses using aggregate_rank and print Rank, Source, Hypothesis, AvgRank (and optionally save/return ranked DataFrame)."""
    if not results:
        print("No results to rank.")
        return None
    ranked_df = rank_hypotheses_with_aggregate(results, metric_specs=metric_specs)
    print(f"\n{'='*100}")
    print("RANKING (aggregate_rank: DetF1, FuzzF1, SurpAUROC, Embed, Judge, p_val, d)")
    print(f"{'='*100}\n")
    display_cols = ["Rank", "Source", "Hypothesis", "AvgRank"]
    print(ranked_df[display_cols].to_string(index=False))
    r_cols = [c for c in ranked_df.columns if c.startswith("r_")]
    if r_cols:
        print(f"\nPer-metric ranks: {', '.join(r_cols)}")
        print(ranked_df[["Rank", "Source", "AvgRank"] + r_cols].to_string(index=False))
    print("\n" + "=" * 100)
    if len(ranked_df) > 0:
        best = ranked_df.iloc[0]
        print(f"🏆 WINNER: {best['Source']} | Rank={best['Rank']:.1f} AvgRank={best['AvgRank']:.2f}")
    print("=" * 100)
    return ranked_df


def detect_language_patterns(activations: List[str]) -> Dict[str, Any]:
    """
    Analyze activation texts to detect language-specific patterns (e.g. French).
    Returns dict with detected_language, confidence, verb/indicator counts, sample words.
    """
    if not activations:
        return {"detected_language": "Unknown/Mixed", "confidence": "LOW", "french_verb_count": 0}
    french_verbs = [
        "trouve", "trouvé", "voudrais", "veux", "as besoin", "veuilles", "imprime", "exécutes",
        "fait", "arriver", "obtenir", "réaliser", "apparaît", "condition", "vide", "décidé",
        "placerais", "tronquer", "avoir", "être", "faire", "pouvoir", "savoir", "vouloir", "devoir",
    ]
    french_articles = ["le", "la", "les", "un", "une", "des", "du", "de la"]
    french_prepositions = ["pour", "par", "avec", "de", "en", "à", "que", "dans", "sur"]
    french_count = 0
    french_verb_count = 0
    for act in activations:
        act_lower = (act or "").lower().replace("\u2581", " ")
        for word in french_verbs:
            if word in act_lower:
                french_verb_count += 1
                french_count += 1
        for word in french_articles + french_prepositions:
            if f" {word} " in act_lower or act_lower.startswith(word + " "):
                french_count += 1
    if french_count >= 3 or french_verb_count >= 2:
        detected_language = "French"
        confidence = "HIGH" if french_verb_count >= 3 else "MEDIUM"
    else:
        detected_language = "Unknown/Mixed"
        confidence = "LOW"
    sample_french_words = [w for w in french_verbs if any(w in (a or "").lower() for a in activations[:5])]
    return {
        "detected_language": detected_language,
        "confidence": confidence,
        "french_verb_count": french_verb_count,
        "french_indicator_count": french_count,
        "sample_french_words": sample_french_words[:10],
    }


def generate_next_hypothesis_from_results(
    tested_results: List[Tuple[str, Any]],
    activating_examples: List[str],
    openai_client: Any,
    model: str = "gpt-4o",
    language_analysis: Optional[Dict[str, Any]] = None,
    temperature: float = 0.8,
) -> str:
    """
    Generate a new hypothesis informed by previous evaluation results and activating examples.
    tested_results: list of (source_label, evaluation) where evaluation has at least
      hypothesis_description (or hypothesis.description) and scores (rank_score, average_rank, effect_size, llm_judge_score, etc.).
    Returns a single-sentence hypothesis string (max ~25 words).
    """
    history_lines = []
    for source, eval_r in tested_results:
        desc = getattr(eval_r, "hypothesis_description", None) or (
            getattr(getattr(eval_r, "hypothesis", None), "description", None) or str(eval_r.get("hypothesis", "")) if isinstance(eval_r, dict) else ""
        )
        if not desc and isinstance(eval_r, dict):
            desc = eval_r.get("hypothesis_description", eval_r.get("hypothesis", ""))
        rank_s = getattr(eval_r, "rank_score", None) or (eval_r.get("rank_score") if isinstance(eval_r, dict) else None)
        avg_r = getattr(eval_r, "average_rank", None) or (eval_r.get("average_rank") if isinstance(eval_r, dict) else None)
        eff = getattr(eval_r, "effect_size", None) or (eval_r.get("effect_size") if isinstance(eval_r, dict) else None)
        llm = getattr(eval_r, "llm_judge_score", None) or (eval_r.get("llm_judge_score") or eval_r.get("llm_rating") if isinstance(eval_r, dict) else None)
        power = getattr(eval_r, "statistical_power", None) or (eval_r.get("statistical_power") if isinstance(eval_r, dict) else None)
        parts = []
        if power is not None:
            parts.append(f"Power={power:.2f}")
        if eff is not None:
            parts.append(f"Effect={eff:.2f}")
        if llm is not None:
            parts.append(f"LLM={llm:.2f}")
        if rank_s is not None:
            parts.append(f"RankScore={rank_s:.2f}")
        if avg_r is not None:
            parts.append(f"AvgRank={avg_r:.2f}")
        history_lines.append(f"[{source}] '{desc[:80]}...' → " + ", ".join(parts))
    history = "\n".join(history_lines)
    if not tested_results:
        best_desc = "none"
        best_parts = "N/A"
    else:
        def _score(x):
            e = x[1]
            p = getattr(e, "statistical_power", None) or (e.get("statistical_power") if isinstance(e, dict) else None)
            rs = getattr(e, "rank_score", None) or (e.get("rank_score") if isinstance(e, dict) else None)
            ar = getattr(e, "average_rank", None) or (e.get("average_rank") if isinstance(e, dict) else None)
            if p is not None:
                return (float(p), -(ar or 999), (rs or 0))
            if ar is not None:
                return (-(ar or 999), rs or 0, 0)
            return (rs or 0, 0, 0)
        best_source, best_eval = max(tested_results, key=_score)
        best_desc = getattr(best_eval, "hypothesis_description", None) or (
            getattr(getattr(best_eval, "hypothesis", None), "description", None) if not isinstance(best_eval, dict) else best_eval.get("hypothesis_description", best_eval.get("hypothesis", ""))
        )
        best_parts = []
        if getattr(best_eval, "statistical_power", None) is not None:
            best_parts.append(f"Statistical Power: {best_eval.statistical_power:.2f}")
        if getattr(best_eval, "effect_size", None) is not None:
            best_parts.append(f"Effect Size: {best_eval.effect_size:.2f}")
        if getattr(best_eval, "llm_judge_score", None) is not None:
            best_parts.append(f"LLM Judge: {best_eval.llm_judge_score:.2f}")
        if getattr(best_eval, "rank_score", None) is not None:
            best_parts.append(f"Rank Score: {best_eval.rank_score:.2f}")
        if isinstance(best_eval, dict):
            if best_eval.get("statistical_power") is not None:
                best_parts.append(f"Statistical Power: {best_eval['statistical_power']:.2f}")
            if best_eval.get("effect_size") is not None:
                best_parts.append(f"Effect Size: {best_eval['effect_size']:.2f}")
            if best_eval.get("llm_judge_score") is not None:
                best_parts.append(f"LLM Judge: {best_eval['llm_judge_score']:.2f}")
        best_parts = "\n".join(best_parts) if best_parts else "N/A"
    activation_analysis = "\n".join([f"  - {act[:100]}..." for act in (activating_examples or [])[:5]])
    language_context = ""
    if language_analysis and language_analysis.get("detected_language") in ("French", "Spanish", "German", "Italian"):
        lang = language_analysis.get("detected_language", "Unknown")
        verb_count = language_analysis.get("french_verb_count") or language_analysis.get("spanish_verb_count") or 0
        sample_words = ", ".join((language_analysis.get("sample_french_words") or language_analysis.get("sample_spanish_words") or [])[:5])
        conf = language_analysis.get("confidence", "UNKNOWN")
        language_context = f"""
CRITICAL LANGUAGE DETECTION:
The top activations are PREDOMINANTLY {lang.upper()} ({verb_count} verbs/indicators detected).
Sample words found: {sample_words}
Confidence: {conf}
Your hypothesis MUST explicitly mention {lang.upper()} and {lang}-specific linguistic features.
"""
    prompt = f"""You are refining hypotheses about an SAE feature in a language model.

TOP ACTIVATING EXAMPLES (the feature fires on these):
{activation_analysis}
{language_context}
TESTED HYPOTHESES SO FAR (with their evaluation scores):
{history}

BEST SO FAR: '{best_desc}'
{best_parts}

CRITICAL: Pay special attention to LANGUAGE-SPECIFIC patterns. If a language is detected, focus on that language's morphology/syntax.

Your task: Generate a NEW hypothesis that:
1. Builds on what worked (high power/effect/rank = good)
2. Addresses weaknesses of previous hypotheses
3. Explores LANGUAGE-SPECIFIC angles if a language was detected
4. Is MORE SPECIFIC than generic descriptions
5. If language is detected, specify which language and what linguistic pattern

Return ONLY a single sentence hypothesis (max 25 words)."""
    try:
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200,
        )
        text = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
        return text
    except Exception as e:
        print(f"⚠️  Error generating next hypothesis: {e}")
        return ""


def generate_new_hypothesis(
    feature_idx: int,
    layer: int,
    feature_data: Optional[pd.DataFrame] = None,
    activations: Optional[List[Dict]] = None,
    logit_info: Optional[Dict] = None,
    openai_client: Optional[Any] = None,
    tested_results: Optional[List[Tuple[str, Any]]] = None,
    activating_examples: Optional[List[str]] = None,
    language_analysis: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o",
    return_string_only: bool = False,
    **kwargs: Any,
) -> Optional[Any]:
    """
    Generate a new hypothesis: either from scratch (Neuronpedia/markers) or informed by previous results.
    - If tested_results and activating_examples are provided, calls generate_next_hypothesis_from_results.
    - Otherwise uses generate_hypothesis_from_neuronpedia (if activations/logit_info) or generate_hypothesis_from_markers (if feature_data).
    Returns FeatureHypothesis or str (if return_string_only=True).
    """
    client = openai_client or kwargs.get("openai_client")
    if not client:
        print("⚠️  generate_new_hypothesis requires openai_client")
        return None
    if tested_results is not None and activating_examples is not None:
        text = generate_next_hypothesis_from_results(
            tested_results, activating_examples, client, model=model or "gpt-4o", language_analysis=language_analysis
        )
        if return_string_only:
            return text
        if text:
            feature_id = f"L{layer}F{feature_idx}"
            return FeatureHypothesis(
                feature_id=feature_id,
                description=text,
                confidence=0.5,
                reasoning="Generated from previous results",
                iteration=len(tested_results) if tested_results else 0,
            )
        return None
    if activations is not None or logit_info is not None:
        acts = activations or []
        logit = logit_info or {}
        hyp = generate_hypothesis_from_neuronpedia(acts, logit, layer, feature_idx, client, model=model)
        return hyp if not return_string_only else (hyp.description if hyp else None)
    if feature_data is not None:
        hyp = generate_hypothesis_from_markers(feature_idx, feature_data, layer, client, model=model)
        return hyp if not return_string_only else (hyp.description if hyp else None)
    print("⚠️  generate_new_hypothesis needs feature_data, or (activations/logit_info), or (tested_results + activating_examples)")
    return None


def save_hypotheses_and_rankings(
    output_dir: Path,
    feature_id: str,
    hypotheses: List[Dict[str, Any]],
    ranked_with_evals: Optional[List[Tuple[str, ComprehensiveEvaluation]]] = None,
    ranked_df: Optional[pd.DataFrame] = None,
    metric_specs: Optional[Dict[str, str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Optional[Path]]:
    """
    Save hypotheses list and optional 7-metric ranking (via aggregate_rank) to JSON and CSV.
    When ranked_with_evals is provided, ranked_df is computed via rank_hypotheses_with_aggregate unless ranked_df is given.
    Returns (json_path, csv_path or None).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = feature_id.replace(" ", "_")
    json_path = output_dir / f"hypotheses_{safe_id}.json"
    data = {
        "feature_id": feature_id,
        "hypotheses": hypotheses,
        "updated_at": pd.Timestamp.now().isoformat(),
        **(metadata or {}),
    }
    if ranked_df is None and ranked_with_evals:
        ranked_df = rank_hypotheses_with_aggregate(ranked_with_evals, metric_specs=metric_specs)
    csv_path = None
    if ranked_df is not None and len(ranked_df) > 0:
        csv_path = output_dir / f"rankings_{safe_id}.csv"
        ranked_df.to_csv(csv_path, index=False)
        data["ranking"] = ranked_df[["Rank", "Source", "Hypothesis", "AvgRank"]].to_dict(orient="records")
        data["ranking_columns"] = list(ranked_df.columns)
        data["seven_metric_ranking"] = ranked_df.to_dict(orient="records")
    if ranked_with_evals and not data.get("seven_metric_ranking"):
        data["seven_metric_ranking"] = [
            {
                "source": src,
                "hypothesis": (e.hypothesis_description or "")[:300],
                "scores": {
                    "DetF1": e.detection_score, "FuzzF1": e.fuzzing_score, "SurpAUROC": e.surprisal_score,
                    "Embed": e.embedding_score, "Judge": e.llm_judge_score, "p_val": e.p_value if e.p_value >= 0.001 else "<0.001", "d": e.effect_size,
                },
            }
            for src, e in ranked_with_evals
        ]

    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)
    return json_path, csv_path


def load_hypotheses_file(path: str) -> Dict[str, Any]:
    """Load a hypotheses JSON file (e.g. hypotheses_L0_F9004.json). Returns dict with feature_id, hypotheses, seven_metric_ranking (if present)."""
    with open(path, "r") as f:
        return json.load(f)


def load_feature_data(results_dir: str) -> Optional[pd.DataFrame]:
    """
    Load feature analysis data from FeatureFinder results.
    
    Args:
        results_dir: Path to FeatureFinder timestamped results directory
        
    Returns:
        DataFrame with feature metadata and statistics
    """
    results_path = Path(results_dir)
    
    # Look for marker features files
    marker_files = list(results_path.glob("top_markers_*_saeL*.csv"))
    
    if not marker_files:
        print(f"⚠️  No marker feature files found in {results_dir}")
        return None
    
    print(f"📁 Found {len(marker_files)} marker feature files")
    
    # Load and combine all marker files
    all_features = []
    for marker_file in marker_files:
        df = pd.read_csv(marker_file)
        
        # Extract category from filename
        # e.g., "french" from "top_markers_french_saeL0_500_prompts"
        category = marker_file.stem.split('_')[2]
        df['category'] = category
        
        # Parse feature_idx from names column (format: "L0_F7288")
        if 'names' in df.columns:
            df['feature_idx'] = df['names'].str.extract(r'F(\d+)')[0].astype(int)
        else:
            print(f"⚠️  Warning: 'names' column not found in {marker_file.name}")
            continue
            
        all_features.append(df)
    
    if all_features:
        combined_df = pd.concat(all_features, ignore_index=True)
        print(f"✓ Loaded {len(combined_df)} marker features across {len(marker_files)} categories")
        return combined_df
    
    return None


def generate_hypothesis_from_markers(
    feature_idx: int,
    feature_data: pd.DataFrame,
    layer: int,
    openai_client,
    model: str = None,
    temperature: float = 0.7
) -> Optional[FeatureHypothesis]:
    """
    Generate initial hypothesis for a feature based on marker analysis.
    
    Args:
        feature_idx: Feature index
        feature_data: DataFrame with feature activation data
        layer: SAE layer number
        openai_client: OpenAI client instance
        model: LLM model to use
        temperature: Sampling temperature
        
    Returns:
        FeatureHypothesis or None
    """
    # Filter data for this specific feature
    feature_rows = feature_data[feature_data['feature_idx'] == feature_idx]
    
    if len(feature_rows) == 0:
        print(f"⚠️  No data found for feature {feature_idx}")
        return None
    
    # Gather information about which categories activate this feature
    categories = feature_rows['category'].unique().tolist()
    
    # Get statistics
    stats_summary = feature_rows.groupby('category').agg({
        'effect_size': 'mean',
        'pvals': 'min'
    }).to_dict()
    
    # Use provided model or default to gpt-4o
    if model is None:
        model = os.environ.get("AGENT_MODEL_NAME", "gpt-4o")
    
    prompt = f"""You are a world-class expert in neural network interpretability, specifically analyzing Sparse Autoencoder (SAE) features from language models.

FEATURE INFORMATION:
- Layer: {layer}
- Feature Index: {feature_idx}
- Strongly activates on categories: {', '.join(categories)}

STATISTICAL MARKERS:
{json.dumps(stats_summary, indent=2)}

This feature is a "marker feature" for these categories, meaning it shows statistically significant activation patterns that distinguish these linguistic/semantic categories.

Generate ONE highly specific, testable hypothesis about what linguistic or semantic pattern this SAE feature detects.

Your hypothesis MUST include:
1. **Precise description** - What specific linguistic/semantic pattern does this feature detect?
2. **Language specificity** - Which language(s) or cross-lingual patterns?
3. **Semantic category** - What type of content (e.g., temporal expressions, negations, technical terms)?
4. **Confidence** - Your honest confidence level (0-1)
5. **Reasoning** - Why would Gemma-2-2B learn to represent this specific pattern?

Return JSON:
{{
  "description": "Ultra-specific one-sentence description of the pattern",
  "language_specificity": "English/French/Cross-lingual/Multilingual/Language-agnostic",
  "semantic_category": "Category like: temporal/spatial/modal/technical/syntactic",
  "confidence": 0.75,
  "reasoning": "Deep analysis of why this pattern is computationally useful for the model"
}}"""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a world-class expert in neural network interpretability."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        
        feature_id = f"L{layer}F{feature_idx}"
        
        return FeatureHypothesis(
            feature_id=feature_id,
            description=data['description'],
            confidence=float(data['confidence']),
            reasoning=data['reasoning'],
            language_specificity=data.get('language_specificity'),
            semantic_category=data.get('semantic_category'),
            iteration=0
        )
    
    except Exception as e:
        print(f"⚠️  Error generating hypothesis: {e}")
        return None


def generate_final_report(
    feature_idx: int,
    layer: int,
    hypothesis: FeatureHypothesis,
    feature_data: pd.DataFrame
) -> FeatureExplanationReport:
    """
    Generate comprehensive explanation report for a feature.
    
    Args:
        feature_idx: Feature index
        layer: SAE layer
        hypothesis: Final hypothesis
        feature_data: DataFrame with feature data
        
    Returns:
        FeatureExplanationReport
    """
    feature_rows = feature_data[feature_data['feature_idx'] == feature_idx]
    top_categories = feature_rows.nlargest(3, 'effect_size')['category'].tolist()
    
    detailed_explanation = f"""
FEATURE: Layer {layer}, Index {feature_idx}

HYPOTHESIS:
{hypothesis.description}

LANGUAGE SPECIFICITY:
{hypothesis.language_specificity}

SEMANTIC CATEGORY:
{hypothesis.semantic_category}

COMPUTATIONAL MOTIVATION:
{hypothesis.reasoning}

TOP ACTIVATING CATEGORIES:
{', '.join(top_categories)}

CONFIDENCE: {hypothesis.confidence:.1%}
"""
    
    return FeatureExplanationReport(
        feature_id=hypothesis.feature_id,
        feature_index=feature_idx,
        layer=layer,
        one_sentence_explanation=hypothesis.description,
        detailed_explanation=detailed_explanation.strip(),
        top_activating_categories=top_categories,
        confidence_score=hypothesis.confidence,
        hypothesis_evolution=[hypothesis.to_dict()]
    )


@dataclass
class TestCase:
    """Test case for validating a hypothesis."""
    text: str
    should_activate: bool
    category: str  # positive, negative, edge_case, adversarial
    rationale: str
    language: Optional[str] = None


@dataclass
class TestResult:
    """Results from testing a single test case."""
    test_case: TestCase
    actual_activation: float
    predicted_correctly: bool
    error_magnitude: float


@dataclass
class Criticism:
    """LLM criticism of hypothesis and test results."""
    strengths: List[str]
    weaknesses: List[str]
    specific_failures: List[str]
    suggested_refinements: List[str]
    overall_assessment: str
    confidence_in_hypothesis: float


def criticism_from_dict(d: Dict[str, Any]) -> Criticism:
    """Build Criticism from a dict (e.g. loaded from JSON)."""
    return Criticism(
        strengths=d.get("strengths", []),
        weaknesses=d.get("weaknesses", []),
        specific_failures=d.get("specific_failures", []),
        suggested_refinements=d.get("suggested_refinements", []),
        overall_assessment=d.get("overall_assessment", ""),
        confidence_in_hypothesis=float(d.get("confidence_in_hypothesis", 0.5)),
    )


def test_results_from_dicts(lst: List[Dict[str, Any]]) -> List[TestResult]:
    """Build list of TestResult from list of dicts (e.g. loaded from JSON)."""
    results = []
    for r in lst or []:
        tc = TestCase(
            text=r.get("text", ""),
            should_activate=bool(r.get("should_activate", False)),
            category=r.get("category", "positive"),
            rationale=r.get("rationale", ""),
            language=r.get("language"),
        )
        results.append(TestResult(
            test_case=tc,
            actual_activation=float(r.get("actual_activation", 0)),
            predicted_correctly=bool(r.get("predicted_correctly", False)),
            error_magnitude=float(r.get("error_magnitude", 0)),
        ))
    return results


class SAEInterface:
    """Interface to load and test SAE features with Gemma-2-2B."""
    
    def __init__(self, layer: int, feature_idx: int, threshold: float = 4.0):
        self.layer = layer
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.sae = None
        self.model = None
        self.tokenizer = None
        self.device = None
        
    def initialize(self):
        """Initialize model and SAE (heavy operation)."""
        print(f"🔧 Loading Gemma-2-2B model and SAE for layer {self.layer}...")
        
        try:
            from sae_lens import SAE
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError as e:
            raise RuntimeError(f"Required library not installed: {e}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  ✓ Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"  ✓ Tokenizer loaded")
        
        print(f"  ⏳ Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b",
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        self.model.eval()
        print(f"  ✓ Model loaded")
        
        sae_id = f"layer_{self.layer}/width_16k/canonical"
        print(f"  ⏳ Loading SAE: {sae_id}...")
        self.sae, self.sae_cfg, self.sae_sparsity = SAE.from_pretrained(
            release="gemma-scope-2b-pt-mlp-canonical",
            sae_id=sae_id,
        )
        self.sae.eval()
        print(f"  ✓ SAE loaded\n")
    
    def get_feature_activation(self, text: str) -> float:
        """Get SAE feature activation for text."""
        if self.model is None or self.sae is None:
            raise RuntimeError("Must call initialize() first")
            
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            return_attention_mask=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        captured_codes = {"codes": None}
        
        def hook_fn(module, input, output):
            x = output[0] if isinstance(output, (tuple, list)) else output
            if self.sae.device != x.device:
                self.sae.to(x.device)
            try:
                with torch.no_grad():
                    x32 = x.to(dtype=torch.float32) if x.dtype != torch.float32 else x
                    codes = self.sae.encode(x32)
                    captured_codes["codes"] = codes.detach().cpu()
            except Exception as e:
                print(f"⚠️  Error encoding: {e}")
        
        try:
            layer_module = self.model.model.layers[self.layer]
            target_module = getattr(layer_module, 'mlp', 
                                  getattr(layer_module, 'feed_forward', None))
            if target_module is None:
                raise RuntimeError(f"No MLP found in layer {self.layer}")
            
            hook = target_module.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = self.model(**inputs)
            hook.remove()
        except Exception as e:
            print(f"⚠️  Error in forward pass: {e}")
            return 0.0
        
        if captured_codes["codes"] is None:
            return 0.0
        
        try:
            codes = captured_codes["codes"]
            feature_activations = codes[0, :, self.feature_idx].numpy()
            return float(np.max(feature_activations))
        except Exception:
            return 0.0
    
    def test_hypothesis(self, test_cases: List[TestCase]) -> Tuple[List[TestResult], Dict]:
        """Test hypothesis using real SAE."""
        results = []
        
        for i, tc in enumerate(test_cases, 1):
            if i % 10 == 0:
                print(f"    Testing case {i}/{len(test_cases)}...")
            
            actual_activation = self.get_feature_activation(tc.text)
            activated = actual_activation > self.threshold
            predicted_correctly = (tc.should_activate == activated)
            
            error_magnitude = max(0, self.threshold - actual_activation) if tc.should_activate else max(0, actual_activation - self.threshold)
            
            results.append(TestResult(
                test_case=tc,
                actual_activation=actual_activation,
                predicted_correctly=predicted_correctly,
                error_magnitude=error_magnitude
            ))
        
        total = len(results)
        correct = sum(1 for r in results if r.predicted_correctly)
        accuracy = correct / total if total > 0 else 0
        
        by_category = {}
        for category in ['positive', 'negative', 'edge_case', 'adversarial']:
            cat_results = [r for r in results if r.test_case.category == category]
            if cat_results:
                cat_correct = sum(1 for r in cat_results if r.predicted_correctly)
                by_category[category] = {
                    'total': len(cat_results),
                    'correct': cat_correct,
                    'accuracy': cat_correct / len(cat_results)
                }
        
        metrics = {
            'total_cases': total,
            'correct_predictions': correct,
            'overall_accuracy': accuracy,
            'by_category': by_category,
            'mean_error_magnitude': sum(r.error_magnitude for r in results) / total if total > 0 else 0
        }
        
        return results, metrics


def generate_test_cases(
    hypothesis: FeatureHypothesis,
    openai_client,
    n_cases: int = 30,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    n_positive: Optional[int] = None,
    n_negative: Optional[int] = None,
    n_edge: Optional[int] = None,
    n_adversarial: Optional[int] = None
) -> List[TestCase]:
    """Generate test cases for hypothesis validation."""
    
    # Determine per-type counts
    if all(v is not None for v in [n_positive, n_negative, n_edge, n_adversarial]):
        pos = int(n_positive)
        neg = int(n_negative)
        edge = int(n_edge)
        adv = int(n_adversarial)
        total = pos + neg + edge + adv
    else:
        pos = n_cases // 3
        neg = n_cases // 3
        edge = n_cases // 6
        adv = n_cases // 6
        total = n_cases

    prompt = f"""Generate {total} rigorous test cases for this hypothesis:

HYPOTHESIS: {hypothesis.description}
LANGUAGE: {hypothesis.language_specificity or 'Any'}
SEMANTIC CATEGORY: {hypothesis.semantic_category}
REASONING: {hypothesis.reasoning}

Generate:
- {pos} POSITIVE (should strongly activate)
- {neg} NEGATIVE (should NOT activate - contrastive)
- {edge} EDGE CASES (boundary conditions)
- {adv} ADVERSARIAL (similar surface features, different semantics)

Each test case:
{{
  "text": "1-3 sentences",
  "should_activate": true/false,
  "category": "positive"/"negative"/"edge_case"/"adversarial",
  "rationale": "Why this tests the hypothesis",
  "language": "Which language"
}}

Return JSON with "test_cases" array."""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in adversarial testing."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=3500,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        
        if not data or 'test_cases' not in data:
            return []
        
        return [
            TestCase(
                text=tc['text'],
                should_activate=tc['should_activate'],
                category=tc['category'],
                rationale=tc['rationale'],
                language=tc.get('language')
            )
            for tc in data['test_cases']
        ]
    
    except Exception as e:
        print(f"⚠️  Error generating test cases: {e}")
        return []


def criticize_hypothesis(
    hypothesis: FeatureHypothesis,
    test_results: List[TestResult],
    metrics: Dict[str, Any],
    openai_client,
    model: str = "gpt-4o",
    temperature: float = 0.5
) -> Optional[Criticism]:
    """Have an LLM critic analyze the hypothesis and results."""
    
    failures = [r for r in test_results if not r.predicted_correctly]
    failure_examples = []
    
    for fail in failures[:10]:
        failure_examples.append(
            f"Expected: {'HIGH' if fail.test_case.should_activate else 'LOW'}, "
            f"Got: {fail.actual_activation:.2f}\n"
            f"Text: {fail.test_case.text}\n"
            f"Rationale: {fail.test_case.rationale}"
        )
    
    by_category_str = json.dumps(metrics.get('by_category', {}), indent=2)
    
    prompt = f"""You are a harsh but fair critic evaluating an SAE feature hypothesis.

HYPOTHESIS:
Description: {hypothesis.description}
Language: {hypothesis.language_specificity}
Category: {hypothesis.semantic_category}
Confidence: {hypothesis.confidence:.0%}
Reasoning: {hypothesis.reasoning}

TEST RESULTS:
- Total cases: {metrics['total_cases']}
- Correct: {metrics['correct_predictions']}
- Accuracy: {metrics['overall_accuracy']:.1%}
- By category: {by_category_str}

FAILURES ({len(failures)} total):
{chr(10).join(failure_examples[:10])}

Provide CRITICAL analysis:
1. What are the hypothesis's strengths?
2. What are its weaknesses?
3. Which specific failures reveal problems?
4. What refinements would improve it?
5. Overall assessment
6. Your confidence in this hypothesis (0-1)

Return JSON:
{{
  "strengths": ["strength 1", "strength 2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "specific_failures": ["failure pattern 1", "failure pattern 2"],
  "suggested_refinements": ["refinement 1", "refinement 2"],
  "overall_assessment": "honest critical assessment",
  "confidence_in_hypothesis": 0.7
}}"""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a critical thinker and harsh but fair evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        
        return Criticism(
            strengths=data.get('strengths', []),
            weaknesses=data.get('weaknesses', []),
            specific_failures=data.get('specific_failures', []),
            suggested_refinements=data.get('suggested_refinements', []),
            overall_assessment=data.get('overall_assessment', ''),
            confidence_in_hypothesis=float(data.get('confidence_in_hypothesis', 0.5))
        )
    
    except Exception as e:
        print(f"⚠️  Error in criticism: {e}")
        return None


def refine_hypothesis(
    hypothesis: FeatureHypothesis,
    criticism: Criticism,
    test_results: List[TestResult],
    openai_client,
    model: str = "gpt-4o",
    temperature: float = 0.6
) -> Optional[FeatureHypothesis]:
    """Refine hypothesis based on criticism and failures."""
    
    refinement_context = f"""
CURRENT HYPOTHESIS:
Description: {hypothesis.description}
Language: {hypothesis.language_specificity}
Category: {hypothesis.semantic_category}
Confidence: {hypothesis.confidence:.0%}

CRITIC'S ASSESSMENT:
Strengths: {', '.join(criticism.strengths)}
Weaknesses: {', '.join(criticism.weaknesses)}
Specific Failures: {', '.join(criticism.specific_failures)}
Suggested Refinements: {', '.join(criticism.suggested_refinements)}
Overall: {criticism.overall_assessment}

REFINEMENT HISTORY:
{chr(10).join(hypothesis.refinement_history) if hypothesis.refinement_history else 'None'}
"""
    
    prompt = f"""{refinement_context}

Based on the critic's feedback, refine this hypothesis to be more accurate.

The refined hypothesis should:
1. Address the weaknesses identified
2. Maintain the strengths
3. Be more specific about edge cases
4. Improve linguistic precision
5. Update confidence appropriately

Return JSON:
{{
  "description": "Refined description",
  "language_specificity": "Updated language info",
  "semantic_category": "Updated or same category",
  "confidence": 0.8,
  "reasoning": "Why this refinement is better",
  "refinement_summary": "One sentence explaining what changed"
}}"""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at refining hypotheses based on evidence."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        
        refined = FeatureHypothesis(
            feature_id=hypothesis.feature_id,
            description=data['description'],
            confidence=float(data['confidence']),
            reasoning=data['reasoning'],
            language_specificity=data.get('language_specificity'),
            semantic_category=data.get('semantic_category'),
            iteration=hypothesis.iteration + 1,
            refinement_history=hypothesis.refinement_history + [data.get('refinement_summary', 'Refinement')]
        )
        
        return refined
    
    except Exception as e:
        print(f"⚠️  Error refining hypothesis: {e}")
        return None


def generate_hypothesis_from_neuronpedia(
    activations: List[Dict],
    logit_info: Dict,
    layer: int,
    feature_idx: int,
    openai_client,
    model: str = None,
    temperature: float = 0.7
) -> Optional[FeatureHypothesis]:
    """
    Generate initial hypothesis from Neuronpedia activation examples.
    This provides richer context than marker statistics alone.
    
    Args:
        activations: List of activation examples from Neuronpedia
        logit_info: Logit information (pos/neg logits)
        layer: SAE layer
        feature_idx: Feature index
        openai_client: OpenAI client
        model: LLM model to use
        temperature: Sampling temperature
        
    Returns:
        FeatureHypothesis or None
    """
    if model is None:
        model = os.environ.get("AGENT_MODEL_NAME", "gpt-4o")
    
    # Format activation examples for LLM (allow empty)
    examples_text = ""
    if activations:
        for i, act in enumerate(activations[:25], 1):
            text_preview = act['text'][:150].replace('\u2581', ' ')
            max_token = act['max_activation_token'].replace('\u2581', ' ')
            examples_text += (
                f"{i}. [Activation: {act['max_activation_value']:.2f}] "
                f"Token: '{max_token}'\n   Context: {text_preview}...\n\n"
            )
    else:
        examples_text = "(No activation examples available on Neuronpedia for this feature – proceeding with logits-only signal.)\n"
    
    # Format logit information
    logit_summary = ""
    if logit_info['top_positive_logits']:
        logit_summary += "TOP POSITIVE LOGITS:\n"
        for item in logit_info['top_positive_logits'][:5]:
            logit_summary += f"  {item['token']}: {item['value']:.3f}\n"
        logit_summary += "\n"
    
    if logit_info['top_negative_logits']:
        logit_summary += "TOP NEGATIVE LOGITS:\n"
        for item in logit_info['top_negative_logits'][:5]:
            logit_summary += f"  {item['token']}: {item['value']:.3f}\n"
    
    prompt = f"""You are a world-class expert in neural network interpretability, specifically analyzing Sparse Autoencoder (SAE) features from Gemma-2-2B.

FEATURE INFORMATION:
- Layer: {layer}
- Feature Index: {feature_idx}

TOP ACTIVATION EXAMPLES (what makes this feature fire):
{examples_text}

LOGIT INFORMATION (what this feature predicts):
{logit_summary}

Using the available Neuronpedia evidence (activation examples if any, and logit tokens), generate ONE highly specific, testable hypothesis about what linguistic or semantic pattern this SAE feature detects.

Your hypothesis MUST include:
1. **Precise description** - What EXACT pattern does this feature detect?
2. **Language specificity** - Which language(s)? Cross-lingual? Language-agnostic?
3. **Semantic category** - What type of content (temporal/spatial/modal/technical/syntactic)?
4. **Confidence** - Your honest confidence level (0-1)
5. **Reasoning** - Why would Gemma-2-2B learn to represent this pattern?

Return JSON:
{{
  "description": "Ultra-specific one-sentence description of the pattern",
  "language_specificity": "English/French/Cross-lingual/Multilingual/Language-agnostic",
  "semantic_category": "temporal/spatial/modal/technical/syntactic/etc",
  "confidence": 0.75,
  "reasoning": "Deep analysis of why this pattern is computationally useful for the model"
}}"""

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a world-class expert in neural network interpretability."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        data = json.loads(response.choices[0].message.content)
        
        feature_id = f"L{layer}F{feature_idx}"
        
        return FeatureHypothesis(
            feature_id=feature_id,
            description=data['description'],
            confidence=float(data['confidence']),
            reasoning=data['reasoning'],
            language_specificity=data.get('language_specificity'),
            semantic_category=data.get('semantic_category'),
            iteration=0
        )
    
    except Exception as e:
        print(f"⚠️  Error generating hypothesis from Neuronpedia: {e}")
        return None
