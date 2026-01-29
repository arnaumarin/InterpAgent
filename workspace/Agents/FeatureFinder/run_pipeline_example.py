#!/usr/bin/env python3
"""
Example script demonstrating how to run the neural atlas pipeline
using the refactored modular structure.

Usage:
    python run_pipeline_example.py [model_key]
    
    where model_key defaults to "gemma_2b"
"""

import sys
import os

# Add workspace to path for imports
workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if workspace_path not in sys.path:
    sys.path.insert(0, workspace_path)

# Import the pipeline class
from Agents.FeatureFinder.modules.featurefinder import CompleteNeuralAtlasPipeline

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        model_key = sys.argv[1]
    else:
        model_key = "gemma_2b"  # Default to Gemma 2B model

    # Configuration
    num_prompts_per_category = 2000  # Dataset size: 2000 prompts per category
    token_statistic = "all_tokens"  # Options: "all_tokens", "mean_last_k", "max_last_k", "top_p_trimmed_mean"
    last_k_tokens = 666  # Number of last tokens to consider
    top_p_trim = 0.5  # Fraction for top-p trimmed mean
    sae_layer_idx = 0  # SAE layer to analyze (default: 0)

    print(f"Starting neural atlas pipeline for {model_key}...")

    # Create pipeline instance
    pipeline = CompleteNeuralAtlasPipeline(
        model_key=model_key,
        max_prompts_per_category=num_prompts_per_category,
        token_statistic=token_statistic,
        last_k_tokens=last_k_tokens,
        top_p_trim=top_p_trim,
        run_permutation_test=False,  # Set to True for rigorous validation
        n_permutations=50,
        sae_layer_idx=sae_layer_idx,
    )

    # Set up directories using model-specific output
    prompts_dir = os.path.join(os.path.dirname(__file__), "multilingual_prompts")
    results_dir = pipeline.output_dir  # For figures and CSVs
    processed_data_dir = pipeline.processed_data_dir  # For large processed data

    # Run complete pipeline
    results, adata, marker_neurons = pipeline.run_complete_pipeline(
        prompts_dir=prompts_dir,
        results_dir=results_dir,
    )

    print(f"\nCOMPLETE NEURAL ATLAS PIPELINE FINISHED FOR {model_key}!")
    print(f"Results saved to: {results_dir}")

