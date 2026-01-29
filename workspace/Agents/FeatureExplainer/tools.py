from langchain_core.tools import tool
from textwrap import dedent


@tool
def load_feature_data_from_results(
    results_dir: str
) -> str:
    """
    Load SAE feature data from FeatureFinder results directory.
    This reads the marker feature CSV files and prepares them for explanation.
    
    Parameters:
        - results_dir: Path to timestamped FeatureFinder results directory 
                      (e.g., /path/to/results_dir/results_gemma2/20251029_180245/)
    
    Returns Python code to execute with python_repl_tool.
    """
    code = f"""
import sys
import os

# Add workspace to path
workspace_root = os.environ.get("WORKSPACE_ROOT", os.path.abspath(os.getcwd()))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from Agents.FeatureExplainer.modules.feature_explainer import load_feature_data

results_dir = r"{results_dir}"

print(f"\n{'='*70}")
print("LOADING FEATURE DATA FROM FEATUREFINDER RESULTS")
print('='*70)
print(f"Results directory: {results_dir}\n")

feature_data = load_feature_data(results_dir)

if feature_data is not None:
    print(f"\n✓ Feature data loaded successfully!")
    print(f"  Total features: {len(feature_data)}")
    print(f"  Unique features: {feature_data['feature_idx'].nunique()}")
    print(f"  Categories: {', '.join(feature_data['category'].unique())}")
    print(f"\n{feature_data.head()}\n")
    
    # Extract layer from results directory name
    import re
    layer_match = re.search(r'saeL(\d+)', str(results_dir))
    if layer_match:
        sae_layer = int(layer_match.group(1))
        print(f"  SAE Layer: {sae_layer}")
    else:
        sae_layer = 0
        print(f"  SAE Layer: {sae_layer} (default)")
else:
    print("❌ Failed to load feature data")

# Please use python_repl_tool to execute this code
"""
    return dedent(code)


@tool
def explain_feature(
    feature_idx: int,
    max_iterations: int = 3,
    n_test_cases: int = 24,
    confidence_threshold: float = 0.85,
    accuracy_threshold: float = 0.80,
    use_agent_model: bool = True,
    cases_per_type: int = 100,
    delete_on_mismatch: bool = False,
    verbose: bool = False,
    save_results: bool = True
) -> str:
    """
    Rigorous SAE feature explanation with hypothesis/test/refine loop.
    
    This runs the COMPLETE iterative refinement process:
    1. Generate initial hypothesis from marker data
    2. Initialize SAE (Gemma-2-2B + SAE weights)
    3. Generate test cases (positive/negative/edge/adversarial)
    4. Test hypothesis with real SAE activations
    5. Get LLM criticism of results
    6. Refine hypothesis based on failures
    7. Repeat until confidence/accuracy thresholds met
    
    Parameters:
        - feature_idx: Feature index to explain
        - max_iterations: Maximum refinement iterations (default: 3)
        - n_test_cases: Number of test cases to generate (default: 24)
        - confidence_threshold: Target confidence level (default: 0.85)
        - accuracy_threshold: Target test accuracy (default: 0.80)
        - use_agent_model: Use same LLM/credentials as the agent (default: True)
                          Automatically uses OPENAI_API_KEY and OPENAI_API_BASE from environment
    
    Requires: feature_data, sae_layer variables from load_feature_data_from_results
    
    Returns Python code to execute with python_repl_tool.
    """
    code = f"""
import openai
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from collections import Counter
import csv

# Verify OpenAI credentials are available
if not os.environ.get("OPENAI_API_KEY"):
    print("\\n" + "="*70)
    print("⚠️  IMPORTANT: OpenAI API Key Required")
    print("="*70)
    print("This tool requires OpenAI API access for:")
    print("  - Hypothesis generation")
    print("  - Test case generation")
    print("  - LLM criticism")
    print("  - Hypothesis refinement")
    print("\\nThe agent should have OPENAI_API_KEY in its environment.")
    print("If you're seeing this message, please contact support.")
    print("="*70 + "\\n")
    raise ValueError("OPENAI_API_KEY not found in environment")

from Agents.FeatureExplainer.modules.feature_explainer import (
    generate_hypothesis_from_markers,
    generate_hypothesis_from_neuronpedia,
    generate_test_cases,
    SAEInterface,
    criticize_hypothesis,
    refine_hypothesis,
    generate_final_report,
    HypothesisEvaluator,
    ComprehensiveEvaluation,
    compute_rankings,
    print_ranking_analysis,
    rank_hypotheses_with_aggregate,
    aggregate_rank,
    METRIC_SPECS_RANK,
)
from Agents.FeatureExplainer.modules.neuronpedia_api import (
    get_top_activations,
    parse_activations,
    extract_logit_info,
    format_layer_for_neuronpedia
)

# Setup OpenAI client - use environment credentials
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_API_BASE")

# Use agent's model if available; fallback to gpt-4o
llm_model = os.environ.get("AGENT_MODEL_NAME") or os.environ.get("OPENAI_MODEL") or "gpt-4o"

print(f"✅ Using OpenAI credentials from environment")
print(f"   Model: {{llm_model}}")
if base_url:
    print(f"   API Base: {{base_url}}")
    openai_client = openai.OpenAI(api_key=api_key, base_url=base_url)
else:
    openai_client = openai.OpenAI(api_key=api_key)

feature_idx = {feature_idx}
max_iterations = {max_iterations}
n_test_cases = {n_test_cases}
confidence_threshold = {confidence_threshold}
accuracy_threshold = {accuracy_threshold}
cases_per_type = {cases_per_type}
delete_on_mismatch = {delete_on_mismatch}
verbose = {verbose}
save_results = {save_results}

# Configure logging verbosity
from builtins import print as _print
print = _print if verbose else (lambda *args, **kwargs: None)

print(f"\\n{{'#'*70}}")
print(f"# SAE FEATURE EXPLANATION WITH TESTING")
print(f"{{'#'*70}}")
print(f"# Feature: Layer {{sae_layer}}, Index {{feature_idx}}")
print(f"# LLM Model: {{llm_model}}")
print(f"# Max iterations: {{max_iterations}}")
total_cases = cases_per_type * 4
print(f"# Test cases per type: {{cases_per_type}} (total {{total_cases}})")
print(f"# Thresholds: Confidence {confidence_threshold:.0%}, Accuracy {accuracy_threshold:.0%}")
print(f"{{'#'*70}}\\n")

# Phase 1: Generate initial hypothesis (try Neuronpedia first, fall back to markers)
print(f"\\n{'='*70}")
print("PHASE 1: TOPIC ALIGNMENT CHECK (Neuronpedia)")
print('='*70)

current_hypothesis = None

# Try Neuronpedia for richer context
print("\\n🌐 Attempting to fetch activation examples from Neuronpedia...")
layer_str = format_layer_for_neuronpedia(sae_layer)
feature_data_np = get_top_activations(layer_str, feature_idx)

if feature_data_np:
    activations = parse_activations(feature_data_np)
    logit_info = extract_logit_info(feature_data_np)
    
    # Early topic alignment check (LLM-as-judge)
    try:
        feature_rows = feature_data[feature_data['feature_idx'] == feature_idx]
        studied_topics = set(feature_rows['category'].astype(str).str.lower().unique().tolist())
        print("Studied topics from FeatureFinder:", sorted(studied_topics))

        topics_list = sorted(list(studied_topics))
        examples_preview = []
        for act in (activations or [])[:12]:
            txt = (act.get('text','') or '')[:240].replace('\u2581',' ')
            examples_preview.append(txt)

        pos_tokens = [it.get('token','') for it in (logit_info.get('top_positive_logits') or [])][:15]
        neg_tokens = [it.get('token','') for it in (logit_info.get('top_negative_logits') or [])][:15]

        judge_payload = {{
            'studied_topics': topics_list,
            'activation_examples': examples_preview,
            'top_positive_tokens': pos_tokens,
            'top_negative_tokens': neg_tokens,
        }}

        judge_prompt = (
            "You are an expert judge determining whether an SAE feature relates to the target topics.\n"
            "Assess RELATION to ANY of the studied topics using: (1) activation text previews, (2) top logit tokens.\n"
            "Return strict JSON with keys: is_related (boolean), score (0-1), explanation (short).\n\n"
            + json.dumps(judge_payload, indent=2)
        )

        resp = openai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": "You are a precise evaluator for topic alignment of SAE features."},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
            max_tokens=300,
            response_format={{"type": "json_object"}}
        )

        try:
            judgment = json.loads(resp.choices[0].message.content)
        except Exception:
            judgment = {{}}

        is_related = bool(judgment.get('is_related', False))
        score = float(judgment.get('score', 0.0))
        explanation = str(judgment.get('explanation', ''))

        print("Topic alignment (LLM-as-judge): is_related=", is_related, "score=", score)
        if explanation:
            print("Reason:", explanation)

        if not is_related:
            _print("\nFeature judged NOT related to the studied topics. Stopping before hypothesis generation.")
            if delete_on_mismatch:
                _print("\nDeletion requested (delete_on_mismatch=True). Removing from top_markers CSVs...")
                results_path = Path(results_dir)
                csv_paths = list(results_path.glob("top_markers_*_saeL*.csv"))
                removed_counts = {{}}
                for csv_path in csv_paths:
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        if 'feature_idx' not in df.columns and 'names' in df.columns:
                            df['feature_idx'] = df['names'].str.extract(r'F(\\d+)')[0].astype(int)
                        before = len(df)
                        df = df[df['feature_idx'] != feature_idx]
                        after = len(df)
                        if after != before:
                            df.to_csv(csv_path, index=False)
                            removed_counts[csv_path.name] = before - after
                    except Exception as e:
                        print(f"   Failed updating {{csv_path.name}}: {{e}}")
                if removed_counts:
                    _print("Updated CSVs (rows removed):")
                    for name, cnt in removed_counts.items():
                        _print(f"   - {{name}}: {{cnt}}")
                else:
                    _print("No matching rows found to remove in marker CSVs.")
            else:
                _print("\nIf you want to remove this feature from marker CSVs, rerun with delete_on_mismatch=True.")
            raise SystemExit(0)
    except Exception as e:
        _print(f"Warning: Early topic alignment LLM check failed: {e}")

    print(f"\n{'='*70}")
    print("PHASE 1B: INITIAL HYPOTHESIS GENERATION")
    print('='*70)
    num_acts = len(activations) if activations else 0
    print(f"Top Activations: {num_acts}")
    if num_acts == 0:
        print("No activation examples on Neuronpedia (\"No Known Activations\"). Using logits-only signal.\n")
    else:
        print(f"   Top activation: {activations[0]['max_activation_value']:.2f}\n")
    
    current_hypothesis = generate_hypothesis_from_neuronpedia(
        activations=activations or [],
        logit_info=logit_info,
        layer=sae_layer,
        feature_idx=feature_idx,
        openai_client=openai_client,
        model=llm_model
    )

# Fall back to marker-based if Neuronpedia unavailable
if not current_hypothesis:
    print("⚠️  Neuronpedia data unavailable, using marker statistics instead\\n")
    current_hypothesis = generate_hypothesis_from_markers(
        feature_idx=feature_idx,
        feature_data=feature_data,
        layer=sae_layer,
        openai_client=openai_client,
        model=llm_model
    )

if not current_hypothesis:
    print("❌ Failed to generate initial hypothesis")
else:
    print(f"\n✓ Initial hypothesis:")
    print(f"  {current_hypothesis.description}")
    print(f"  Confidence: {current_hypothesis.confidence:.0%}")
    
    # Early validation: ensure hypothesis aligns with FeatureFinder categories
    try:
        feature_rows = feature_data[feature_data['feature_idx'] == feature_idx]
        marker_categories = set(
            feature_rows['category'].astype(str).str.lower().unique().tolist()
        )
        hyp_lang = (current_hypothesis.language_specificity or '').lower()
        hyp_sem = (current_hypothesis.semantic_category or '').lower()
        hyp_desc = (current_hypothesis.description or '').lower()
        cross_terms = {"cross-lingual", "multilingual", "language-agnostic", "any", "all"}
        is_lang_related = any(cat in hyp_lang for cat in marker_categories) or (
            hyp_lang in cross_terms and len(marker_categories) > 1
        )
        is_sem_related = any(cat in hyp_sem for cat in marker_categories) or any(
            cat in hyp_desc for cat in marker_categories
        )
        is_related = is_lang_related or is_sem_related
        print("\nCategory alignment check:")
        print(f"  FeatureFinder categories: {{sorted(marker_categories)}}")
        print(f"  Hypothesis language: '{{current_hypothesis.language_specificity}}' | semantic: '{{current_hypothesis.semantic_category}}'")
        print(f"  Related: {{'YES' if is_related else 'NO'}}")
        if not is_related:
            print("\nHypothesis not aligned with FeatureFinder categories.")
            if delete_on_mismatch:
                print("Deletion requested (delete_on_mismatch=True). Removing from top_markers CSVs...")
                results_path = Path(results_dir)
                csv_paths = list(results_path.glob("top_markers_*_saeL*.csv"))
                removed_counts = {{}}
                for csv_path in csv_paths:
                    try:
                        import pandas as pd
                        df = pd.read_csv(csv_path)
                        if 'feature_idx' not in df.columns and 'names' in df.columns:
                            df['feature_idx'] = df['names'].str.extract(r'F(\\d+)')[0].astype(int)
                        before = len(df)
                        df = df[df['feature_idx'] != feature_idx]
                        after = len(df)
                        if after != before:
                            df.to_csv(csv_path, index=False)
                            removed_counts[csv_path.name] = before - after
                    except Exception as e:
                        print(f"   Failed updating {{csv_path.name}}: {{e}}")
                if removed_counts:
                    print("Updated CSVs (rows removed):")
                    for name, cnt in removed_counts.items():
                        print(f"   - {{name}}: {{cnt}}")
                else:
                    print("No matching rows found to remove in marker CSVs.")
            else:
                print("No deletion performed. If you want to remove this feature from marker CSVs, rerun with delete_on_mismatch=True.")
            # Optionally save rejection record
            if save_results:
                output_dir = Path(results_dir) / "feature_explanations"
                output_dir.mkdir(exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = output_dir / f"feature_L{{sae_layer}}_F{{feature_idx}}_{{timestamp}}_rejected.json"
                with open(output_file, 'w') as f:
                    json.dump({
                        'feature_id': f"L{sae_layer}F{feature_idx}",
                        'status': 'rejected_due_to_mismatch',
                        'marker_categories': sorted(marker_categories),
                        'hypothesis': current_hypothesis.to_dict(),
                    }, f, indent=2)
                print(f"Rejection saved to: {{output_file}}")
            print("Aborting explanation for this feature due to mismatch.")
            raise SystemExit(0)
    except Exception as e:
        print(f"⚠️  Category alignment check failed: {e}")
    
    all_hypotheses = [current_hypothesis]
    iteration_metrics = []  # Track metrics for each iteration
    
    # Phase 2: Initialize SAE
    print(f"\\n{'='*70}")
    print("PHASE 2: INITIALIZING SAE")
    print('='*70)
    
    sae_interface = SAEInterface(layer=sae_layer, feature_idx=feature_idx)
    sae_interface.initialize()
    
    # Phase 3: Iterative testing and refinement
    iteration = 0
    final_metrics = None
    total_test_cases_run = 0
    total_correct_predictions = 0
    
    print(f"\\n{'='*70}")
    print("OPTIMIZATION TRAJECTORY")
    print('='*70)
    
    while iteration < max_iterations:
        iteration += 1
        
        print(f"\\n{'='*70}")
        print(f"PHASE 3.{{iteration}}: TEST → CRITICIZE → REFINE (Iteration {{iteration}}/{{max_iterations}})")
        print('='*70)
        
        # Extract short description for trajectory display
        hyp_desc_short = current_hypothesis.description[:60] + "..." if len(current_hypothesis.description) > 60 else current_hypothesis.description
        
        print(f"\\nCurrent Hypothesis (Iteration {{current_hypothesis.iteration}}):")
        print(f"  {{current_hypothesis.description}}\\n")
        
        # Generate test cases using agent's model
        print("📝 Generating test cases...")
        test_cases = generate_test_cases(
            current_hypothesis,
            openai_client,
            n_cases=total_cases,
            model=llm_model,
            n_positive=cases_per_type,
            n_negative=cases_per_type,
            n_edge=cases_per_type,
            n_adversarial=cases_per_type
        )
        
        if not test_cases:
            print("⚠️  Failed to generate test cases, stopping")
            break
        
        print(f"✓ Generated {{len(test_cases)}} test cases")
        comp = Counter(tc.category for tc in test_cases)
        print(f"   Composition → +: {{comp.get('positive', 0)}}, -: {{comp.get('negative', 0)}}, edge: {{comp.get('edge_case', 0)}}, adv: {{comp.get('adversarial', 0)}}")
        
        # Test hypothesis
        print("⚙️  Testing with SAE...")
        test_results, metrics = sae_interface.test_hypothesis(test_cases)
        final_metrics = metrics
        
        # Store metrics for this iteration BEFORE refinement
        iteration_metrics.append({{
            'iteration': iteration,
            'hypothesis_iteration': current_hypothesis.iteration,
            'description': current_hypothesis.description,
            'description_short': hyp_desc_short,
            'test_accuracy': metrics['overall_accuracy'],
            'confidence': current_hypothesis.confidence,
            'test_cases': metrics['total_cases'],
            'correct_predictions': metrics['correct_predictions'],
            'by_category': metrics.get('by_category', {{}})
        }})
        
        print(f"\n📊 TEST RESULTS:")
        print(f"  Overall Accuracy: {{metrics['overall_accuracy']:.1%}} ({{metrics['correct_predictions']}}/{{metrics['total_cases']}})")
        total_test_cases_run += metrics['total_cases']
        total_correct_predictions += metrics['correct_predictions']
        if total_test_cases_run > 0:
            cum_acc = total_correct_predictions / total_test_cases_run
            print(f"  Cumulative Accuracy: {{cum_acc:.1%}} ({{total_correct_predictions}}/{{total_test_cases_run}})")
        
        if metrics['by_category']:
            print(f"  By Category:")
            for cat, cat_metrics in metrics['by_category'].items():
                acc = cat_metrics['accuracy']
                emoji = "✓" if acc > 0.8 else "⚠️" if acc > 0.6 else "✗"
                print(f"    {{emoji}} {{cat:12s}}: {{acc:.1%}} ({{cat_metrics['correct']}}/{{cat_metrics['total']}})")
        
        # Display trajectory update
        print(f"\\n📈 TRAJECTORY UPDATE:")
        print(f"  Turn {{iteration}} (Hypothesis: {{hyp_desc_short}}):")
        print(f"    Score = {{metrics['overall_accuracy']:.2f}} (Accuracy: {{metrics['overall_accuracy']:.1%}}, Confidence: {{current_hypothesis.confidence:.1%}})")
        
        # Criticize using agent's model
        print("\\n💬 Getting LLM criticism...")
        criticism = criticize_hypothesis(
            current_hypothesis,
            test_results,
            metrics,
            openai_client,
            model=llm_model
        )
        
        if criticism:
            print(f"\\n💬 CRITIC'S ASSESSMENT:")
            print(f"  Confidence in hypothesis: {{criticism.confidence_in_hypothesis:.0%}}")
            print(f"  Strengths: {{len(criticism.strengths)}}")
            for s in criticism.strengths[:3]:
                print(f"    ✓ {{s}}")
            print(f"  Weaknesses: {{len(criticism.weaknesses)}}")
            for w in criticism.weaknesses[:3]:
                print(f"    ✗ {{w}}")
        
        # Check if we've succeeded
        if (metrics['overall_accuracy'] >= accuracy_threshold and 
            current_hypothesis.confidence >= confidence_threshold):
            
            print(f"\\n🎉 SUCCESS! High confidence achieved!")
            print(f"   Accuracy: {{metrics['overall_accuracy']:.1%}}")
            print(f"   Confidence: {{current_hypothesis.confidence:.0%}}")
            break
        
        # Refine if not last iteration using agent's model
        if iteration < max_iterations and criticism:
            print("\\n✨ Refining hypothesis...")
            refined_hypothesis = refine_hypothesis(
                current_hypothesis,
                criticism,
                test_results,
                openai_client,
                model=llm_model
            )
            
            if refined_hypothesis:
                print(f"  Old: {{current_hypothesis.description}}")
                print(f"  New: {{refined_hypothesis.description}}")
                print(f"  Confidence: {{current_hypothesis.confidence:.0%}} → {{refined_hypothesis.confidence:.0%}}")
                print(f"  Hypotheses tested so far: {{len(all_hypotheses)+1}}")
                
                current_hypothesis = refined_hypothesis
                all_hypotheses.append(current_hypothesis)
            else:
                print("⚠️  Refinement failed")
    
    # Phase 3.5: Comprehensive evaluation and ranking (7 metrics)
    ranked_hypotheses_with_evals = []
    if all_hypotheses and final_metrics and iteration_metrics:
        print(f"\\n{'='*70}")
        print("PHASE 3.5: COMPREHENSIVE EVALUATION & RANKING (7 Metrics)")
        print('='*70)
        # Build activating / non-activating examples and scores from last test run
        activating_texts = [r.test_case.text for r in test_results if r.test_case.should_activate]
        non_activating_texts = [r.test_case.text for r in test_results if not r.test_case.should_activate]
        activating_scores = np.array([r.actual_activation for r in test_results if r.test_case.should_activate], dtype=float)
        non_activating_scores = np.array([r.actual_activation for r in test_results if not r.test_case.should_activate], dtype=float)
        if len(activating_scores) < 2:
            activating_scores = np.array([0.0, 0.0])
        if len(non_activating_scores) < 2:
            non_activating_scores = np.array([0.0, 0.0])
        if activating_texts and non_activating_texts:
            evaluator = HypothesisEvaluator(openai_client, model=llm_model)
            for h in all_hypotheses:
                ev = evaluator.evaluate(
                    hypothesis=h.description,
                    activating_examples=activating_texts,
                    non_activating_examples=non_activating_texts,
                    activating_scores=activating_scores,
                    non_activating_scores=non_activating_scores,
                    verbose=verbose,
                )
                ranked_hypotheses_with_evals.append((f"iter{{h.iteration}}", ev))
            if ranked_hypotheses_with_evals:
                print_ranking_analysis(ranked_hypotheses_with_evals)
        else:
            print("  Skipped (need both activating and non-activating examples from test run)")
    
    # Phase 4: Generate final report (after all iterations)
    print(f"\\n{'='*70}")
    print("PHASE 4: FINAL REPORT")
    print('='*70)
    
    # Display complete optimization trajectory
    print(f"\\n{'='*70}")
    print("COMPLETE OPTIMIZATION TRAJECTORY")
    print('='*70)
    for i, iter_metric in enumerate(iteration_metrics, 1):
        desc_short = iter_metric['description_short']
        score = iter_metric['test_accuracy']
        conf = iter_metric['confidence']
        print(f"Turn {{i}} (Hypothesis: {{desc_short}}): Score = {{score:.2f}} (Accuracy: {{score:.1%}}, Confidence: {{conf:.1%}})")
    print('='*70)
    
    report = generate_final_report(
        feature_idx=feature_idx,
        layer=sae_layer,
        hypothesis=current_hypothesis,
        feature_data=feature_data
    )
    
    # Add test metrics to report
    report.test_accuracy = final_metrics['overall_accuracy'] if final_metrics else 0.0
    
    # Capture last criticism and test_results for refine_hypothesis tool
    last_criticism_dict = None
    last_test_results_list = None
    if criticism:
        from dataclasses import asdict
        last_criticism_dict = asdict(criticism)
    if test_results:
        last_test_results_list = [
            {{'text': r.test_case.text, 'should_activate': r.test_case.should_activate, 'actual_activation': r.actual_activation,
              'predicted_correctly': r.predicted_correctly, 'category': r.test_case.category, 'rationale': getattr(r.test_case, 'rationale', '') or '', 'error_magnitude': r.error_magnitude}}
            for r in test_results
        ]
    
    # Optionally save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(results_dir) / "feature_explanations"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"feature_L{{sae_layer}}_F{{feature_idx}}_{{timestamp}}.json"
        results_dict = {{
            'feature_id': report.feature_id,
            'final_hypothesis': current_hypothesis.to_dict(),
            'all_hypotheses': [h.to_dict() for h in all_hypotheses],
            'test_accuracy': report.test_accuracy,
            'optimization_trajectory': iteration_metrics,
            'report': report.to_dict(),
            'last_criticism': last_criticism_dict,
            'last_test_results': last_test_results_list,
        }}
        if ranked_hypotheses_with_evals:
            ranked_df = rank_hypotheses_with_aggregate(ranked_hypotheses_with_evals)
            results_dict['ranking_table'] = ranked_df.to_dict(orient='records')
            results_dict['seven_metric_ranking'] = [
                {{'source': src, 'hypothesis': e.hypothesis_description[:200], 'average_rank': e.average_rank,
                  'dominance_score': e.dominance_score, 'rank_score': e.rank_score,
                  'scores': {{'detection': e.detection_score, 'fuzzing': e.fuzzing_score, 'surprisal': e.surprisal_score,
                             'embedding': e.embedding_score, 'p_value': e.p_value, 'effect_size': e.effect_size, 'llm_judge': e.llm_judge_score}}}}
                for src, e in ranked_hypotheses_with_evals
            ]
        def _json_default(obj):
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {{type(obj)}} is not JSON serializable")
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=_json_default)
        print(f"Saved results to: {{output_file}}")
        # Append to summary CSV (feature_id, explanation, confidence)
        summary_csv = output_dir / "explanations_summary.csv"
        try:
            csv_exists = summary_csv.exists()
            with open(summary_csv, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if not csv_exists:
                    writer.writerow(["feature_id", "explanation", "confidence"]) 
                writer.writerow([report.feature_id, current_hypothesis.description, f"{current_hypothesis.confidence:.4f}"])
            print(f"Updated summary CSV: {{summary_csv}}")
        except Exception as e:
            print(f"Failed updating summary CSV: {{e}}")
    
    # Emit a concise natural-language explanation
    _print("\n" + "="*70)
    _print("SAE Feature Explanation - Final Summary")
    _print("="*70)
    _print(f"Feature: L{{sae_layer}} F{{feature_idx}}")
    _print(f"Description: {{current_hypothesis.description}}")
    _print(f"Language specificity: {{current_hypothesis.language_specificity}} | Category: {{current_hypothesis.semantic_category}}")
    _print(f"Confidence: {{current_hypothesis.confidence:.1%}} | Test accuracy: {{report.test_accuracy:.1%}}")
    _print(f"Iterations: {{iteration}} | Hypotheses tested: {{len(all_hypotheses)}} | Total test cases: {{total_test_cases_run}}")
    _print("\\nOptimization Trajectory:")
    for i, iter_metric in enumerate(iteration_metrics, 1):
        desc_short = iter_metric['description_short']
        score = iter_metric['test_accuracy']
        _print(f"  Turn {{i}}: {{desc_short}} → Score = {{score:.2f}}")
    _print("="*70 + "\n")

# Please use python_repl_tool to execute this code
"""
    return dedent(code)


@tool
def generate_hypothesis(
    feature_idx: int,
    results_dir: str,
    use_previous_results: bool = False,
    from_neuronpedia: bool = True,
    save_results: bool = True,
) -> str:
    """
    Generate a new hypothesis for the feature.
    - If use_previous_results=True: loads saved hypotheses/rankings and generates a NEW hypothesis informed by them (and by top activations).
    - Otherwise: generates from Neuronpedia (if from_neuronpedia=True) or from marker data (feature_data must exist from load_feature_data_from_results).
    Saves the hypothesis list to results_dir/feature_explanations/hypotheses_L{layer}_F{idx}.json when save_results=True.
    Requires: sae_layer (and optionally feature_data) from load_feature_data_from_results. Uses OPENAI_API_KEY.
    Returns Python code to execute with python_repl_tool.
    """
    code = f'''
import sys
import os
import json
from pathlib import Path

workspace_root = os.environ.get("WORKSPACE_ROOT", os.path.abspath(os.getcwd()))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import openai
from Agents.FeatureExplainer.modules.feature_explainer import (
    generate_new_hypothesis,
    generate_next_hypothesis_from_results,
    load_hypotheses_file,
    save_hypotheses_and_rankings,
    detect_language_patterns,
)
from Agents.FeatureExplainer.modules.neuronpedia_api import (
    get_top_activations,
    parse_activations,
    extract_logit_info,
    format_layer_for_neuronpedia,
)

feature_idx = {feature_idx}
results_dir = r"{results_dir}"
use_previous_results = {str(use_previous_results)}
from_neuronpedia = {str(from_neuronpedia)}
save_results = {str(save_results)}

# sae_layer from load_feature_data_from_results
sae_layer = sae_layer if 'sae_layer' in dir() else 0
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_API_BASE")
openai_client = openai.OpenAI(api_key=api_key, base_url=base_url) if api_key else None
llm_model = os.environ.get("AGENT_MODEL_NAME") or "gpt-4o"

output_dir = Path(results_dir) / "feature_explanations"
output_dir.mkdir(parents=True, exist_ok=True)
feature_id = f"L{{sae_layer}}F{{feature_idx}}"
hypotheses_file = output_dir / f"hypotheses_{{feature_id.replace(chr(32), '_')}}.json"

tested_results = []
activating_examples = []
language_analysis = None

if use_previous_results and hypotheses_file.exists():
    data = load_hypotheses_file(str(hypotheses_file))
    hypotheses_list = data.get("hypotheses", [])
    if data.get("seven_metric_ranking"):
        for r in data["seven_metric_ranking"]:
            src = r.get("Source") or r.get("source", "")
            hyp = r.get("Hypothesis") or r.get("hypothesis", "")
            s = r.get("scores") or {{}}
            avg_r = r.get("AvgRank") if "AvgRank" in r else r.get("average_rank")
            # Rank from DataFrame is 1=best; rank_score must be higher=better, so invert when using Rank
            raw_rank = r.get("Rank") if "Rank" in r else None
            rank_s = (1.0 / raw_rank if raw_rank and raw_rank > 0 else None) if raw_rank is not None else r.get("rank_score")
            eff = r.get("d") if "d" in r else s.get("effect_size")
            llm = r.get("Judge") if "Judge" in r else s.get("llm_judge")
            e = type("Eval", (), dict(hypothesis_description=hyp, rank_score=rank_s, average_rank=avg_r, effect_size=eff, llm_judge_score=llm))()
            tested_results.append((src, e))
else:
    hypotheses_list = []

# Get activating examples (Neuronpedia or fallback)
if from_neuronpedia and openai_client:
    layer_str = format_layer_for_neuronpedia(sae_layer)
    raw = get_top_activations(layer_str, feature_idx)
    if raw:
        acts = parse_activations(raw)
        activating_examples = [a.get("text", "")[:500] for a in (acts or [])[:15]]
        language_analysis = detect_language_patterns(activating_examples)
if not activating_examples and "feature_data" in dir():
    import pandas as pd
    fr = feature_data[feature_data["feature_idx"] == feature_idx]
    if len(fr) > 0:
        activating_examples = fr.head(10).get("text", pd.Series()).tolist() if "text" in fr.columns else []
if not activating_examples:
    activating_examples = ["Example activating text 1", "Example 2"]

# Generate
if use_previous_results and tested_results and activating_examples and openai_client:
    from Agents.FeatureExplainer.modules.feature_explainer import generate_next_hypothesis_from_results
    new_text = generate_next_hypothesis_from_results(
        tested_results, activating_examples, openai_client, model=llm_model, language_analysis=language_analysis
    )
    print(f"Generated (informed): {{new_text}}")
    hypotheses_list.append({{"source": f"Generated-{{len(hypotheses_list)}}", "hypothesis": new_text, "description": new_text}})
else:
    hyp = generate_new_hypothesis(
        feature_idx=feature_idx, layer=sae_layer, feature_data=feature_data if "feature_data" in dir() else None,
        activations=parse_activations(get_top_activations(format_layer_for_neuronpedia(sae_layer), feature_idx)) if from_neuronpedia else None,
        logit_info=extract_logit_info(get_top_activations(format_layer_for_neuronpedia(sae_layer), feature_idx)) if from_neuronpedia else None,
        openai_client=openai_client, model=llm_model
    )
    if hyp:
        print(f"Generated: {{hyp.description}}")
        hypotheses_list.append({{"source": "Initial", "hypothesis": hyp.description, "description": hyp.description, **hyp.to_dict()}})
    else:
        print("No hypothesis generated")

if save_results and hypotheses_list:
    json_path, _ = save_hypotheses_and_rankings(output_dir, feature_id, hypotheses_list)
    print(f"Saved to {{json_path}}")
# Please use python_repl_tool to execute this code
'''
    return dedent(code)


@tool
def refine_hypothesis(
    feature_idx: int,
    results_dir: str,
    save_results: bool = True,
) -> str:
    """
    Refine the current (last) hypothesis using the last criticism and test results from a previous explain_feature run.
    Loads the most recent feature explanation JSON from results_dir/feature_explanations/ that contains last_criticism and last_test_results,
    refines the last hypothesis, appends the refined hypothesis to the list, and saves.
    Requires: sae_layer. Uses OPENAI_API_KEY.
    Returns Python code to execute with python_repl_tool.
    """
    code = f'''
import sys
import os
import json
from pathlib import Path

workspace_root = os.environ.get("WORKSPACE_ROOT", os.path.abspath(os.getcwd()))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import openai
from Agents.FeatureExplainer.modules.feature_explainer import (
    refine_hypothesis as refine_hypothesis_fn,
    criticism_from_dict,
    test_results_from_dicts,
    load_hypotheses_file,
    save_hypotheses_and_rankings,
    FeatureHypothesis,
)

feature_idx = {feature_idx}
results_dir = r"{results_dir}"
save_results = {str(save_results)}

sae_layer = sae_layer if "sae_layer" in dir() else 0
api_key = os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("OPENAI_API_BASE")
openai_client = openai.OpenAI(api_key=api_key, base_url=base_url) if api_key else None
llm_model = os.environ.get("AGENT_MODEL_NAME") or "gpt-4o"

output_dir = Path(results_dir) / "feature_explanations"
feature_id = f"L{{sae_layer}}F{{feature_idx}}"

# Find latest feature explanation JSON that has last_criticism and last_test_results
explanation_files = sorted(output_dir.glob("feature_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
loaded = None
for path in explanation_files:
    if str(feature_idx) in path.name and ("L" + str(sae_layer)) in path.name:
        with open(path) as f:
            data = json.load(f)
        if data.get("last_criticism") and data.get("last_test_results"):
            loaded = data
            break
if not loaded:
    print("No prior explain_feature run found with last_criticism and last_test_results. Run explain_feature first.")
else:
    all_hypotheses_dicts = loaded.get("all_hypotheses", [])
    if not all_hypotheses_dicts:
        print("No hypotheses in file.")
    else:
        last_h = all_hypotheses_dicts[-1]
        current = FeatureHypothesis(
            feature_id=last_h.get("feature_id", feature_id),
            description=last_h.get("description", last_h.get("hypothesis", "")),
            confidence=float(last_h.get("confidence", 0.5)),
            reasoning=last_h.get("reasoning", ""),
            language_specificity=last_h.get("language_specificity"),
            semantic_category=last_h.get("semantic_category"),
            iteration=last_h.get("iteration", 0),
            refinement_history=last_h.get("refinement_history", []),
        )
        criticism = criticism_from_dict(loaded["last_criticism"])
        test_results = test_results_from_dicts(loaded["last_test_results"])
        refined = refine_hypothesis_fn(current, criticism, test_results, openai_client, model=llm_model)
        if refined:
            print(f"Refined: {{refined.description}}")
            hypotheses_list = [{{"source": f"iter{{i}}", "hypothesis": h.get("description", h.get("hypothesis", "")), **h}} for i, h in enumerate(all_hypotheses_dicts)]
            hypotheses_list.append({{"source": f"refined-{{len(hypotheses_list)}}", "hypothesis": refined.description, "description": refined.description, **refined.to_dict()}})
            if save_results:
                json_path, _ = save_hypotheses_and_rankings(output_dir, feature_id, hypotheses_list)
                print(f"Saved to {{json_path}}")
        else:
            print("Refinement failed")
# Please use python_repl_tool to execute this code
'''
    return dedent(code)


@tool
def rank_hypotheses(
    feature_idx: int,
    results_dir: str,
    hypotheses_file: str = "",
    save_ranking: bool = True,
) -> str:
    """
    Rank all hypotheses for the feature using the 7-metric evaluation (Detection, Fuzzing, Surprisal, Embedding, P-value, Cohen's d, LLM Judge).
    Loads hypotheses from results_dir/feature_explanations/hypotheses_L{layer}_F{idx}.json (or from hypotheses_file path).
    Fetches activating/non-activating examples from Neuronpedia (and generic negatives), runs HypothesisEvaluator on each hypothesis, computes rankings, prints analysis, and saves to rankings_*.csv and updates the hypotheses JSON with seven_metric_ranking when save_ranking=True.
    Requires: sae_layer. Uses OPENAI_API_KEY.
    Returns Python code to execute with python_repl_tool.
    """
    code = f'''
import sys
import os
import numpy as np
from pathlib import Path

workspace_root = os.environ.get("WORKSPACE_ROOT", os.path.abspath(os.getcwd()))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

import openai
from Agents.FeatureExplainer.modules.feature_explainer import (
    HypothesisEvaluator,
    ComprehensiveEvaluation,
    compute_rankings,
    print_ranking_analysis,
    load_hypotheses_file,
    save_hypotheses_and_rankings,
)
from Agents.FeatureExplainer.modules.neuronpedia_api import get_top_activations, parse_activations, format_layer_for_neuronpedia

feature_idx = {feature_idx}
results_dir = r"{results_dir}"
hypotheses_file_arg = r"{hypotheses_file}"
save_ranking = {str(save_ranking)}

sae_layer = sae_layer if "sae_layer" in dir() else 0
output_dir = Path(results_dir) / "feature_explanations"
feature_id = f"L{{sae_layer}}F{{feature_idx}}"
hyp_path = Path(hypotheses_file_arg) if hypotheses_file_arg else output_dir / f"hypotheses_{{feature_id.replace(chr(32), '_')}}.json"

if not hyp_path.exists():
    print(f"Hypotheses file not found: {{hyp_path}}. Run generate_hypothesis or explain_feature first.")
else:
    data = load_hypotheses_file(str(hyp_path))
    hypotheses_list = data.get("hypotheses", [])
    if not hypotheses_list:
        print("No hypotheses to rank.")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_API_BASE")
        openai_client = openai.OpenAI(api_key=api_key, base_url=base_url) if api_key else None
        llm_model = os.environ.get("AGENT_MODEL_NAME") or "gpt-4o"
        evaluator = HypothesisEvaluator(openai_client, model=llm_model)
        layer_str = format_layer_for_neuronpedia(sae_layer)
        raw = get_top_activations(layer_str, feature_idx)
        acts = parse_activations(raw) if raw else []
        activating_texts = [a.get("text", "")[:500] for a in (acts or [])[:15]]
        non_activating_texts = [
            "The weather today is sunny.", "I went to the store.", "The book is about history.",
            "Technology continues to advance.", "She enjoys music.",
        ]
        if len(activating_texts) < 2:
            activating_texts = activating_texts + ["Example activating 1", "Example activating 2"]
        activating_scores = np.array([1.0] * len(activating_texts[:10]), dtype=float)
        non_activating_scores = np.array([0.1] * len(non_activating_texts[:10]), dtype=float)
        ranked_with_evals = []
        for i, h in enumerate(hypotheses_list):
            desc = h.get("description", h.get("hypothesis", ""))
            src = h.get("source", f"hyp{{i}}")
            ev = evaluator.evaluate(desc, activating_texts, non_activating_texts, activating_scores, non_activating_scores, verbose=True)
            ranked_with_evals.append((src, ev))
        print_ranking_analysis(ranked_with_evals)
        if save_ranking:
            json_path, csv_path = save_hypotheses_and_rankings(output_dir, feature_id, hypotheses_list, ranked_with_evals=ranked_with_evals)
            print(f"Saved ranking to {{json_path}} and {{csv_path}}")
# Please use python_repl_tool to execute this code
'''
    return dedent(code)
