from langchain_core.tools import tool
from textwrap import dedent
import os
import sys

@tool
def environment_setup_tool(
    workspace_root: str,
    prompts_dir: str = "",
    concepts: str = "",
    prompt_files: str = ""
) -> str:
    """
    Sets up the Python environment and validates directories for the Neural Atlas Pipeline.
    Imports all necessary modules and configures the workspace path.

    Parameters:
        - workspace_root: Root path of the workspace (e.g. current dir if run from workspace, or set WORKSPACE_ROOT)
        - prompts_dir: Path to prompts folder (optional if prompt_files is provided)
        - concepts: Comma-separated list of concepts to analyze (e.g., "english,chinese,french")
                    Leave empty to analyze all available concepts.
        - prompt_files: Alternative to concepts - specify exact prompt files with categories
                       Format: "category1:/path/to/file1.txt,category2:/path/to/file2.txt"
                       Example: "english:/path/custom_en.txt,chinese:/path/custom_zh.txt"
                       Takes precedence over concepts parameter.

    Available concepts: english, german, french, italian, spanish, portuguese, chinese, japanese, indonesian

    Expected prompts_dir structure (when using concepts):
        prompts_dir/
            ├── prompts_english.txt
            ├── prompts_german.txt
            ├── prompts_french.txt
            ├── prompts_italian.txt
            ├── prompts_spanish.txt
            ├── prompts_portuguese.txt
            ├── prompts_chinese.txt
            ├── prompts_japanese.txt
            └── prompts_indonesian.txt
    
    Please use python_repl_tool to execute this code directly.
    Do not modify this code arbitrarily.
    """
    code = f"""
        import os
        import sys
        import json
        import pickle
        import warnings
        warnings.filterwarnings('ignore')

        WORKSPACE_ROOT = r\"\"\"{workspace_root}\"\"\"
        if WORKSPACE_ROOT not in sys.path:
            sys.path.insert(0, WORKSPACE_ROOT)
            print(f"✓ Added to Python path: {{WORKSPACE_ROOT}}")

        PROMPTS_DIR = r\"\"\"{prompts_dir}\"\"\"
        CONCEPTS_STR = {concepts!r}
        PROMPT_FILES_STR = {prompt_files!r}

        # Parse prompt files or concepts
        prompt_files_dict = None
        selected_concepts = None

        if PROMPT_FILES_STR:
            # Parse custom prompt files: "category1:/path/file1.txt,category2:/path/file2.txt"
            prompt_files_dict = {{}}
            for pair in PROMPT_FILES_STR.split(','):
                if ':' in pair:
                    category, filepath = pair.split(':', 1)
                    prompt_files_dict[category.strip()] = filepath.strip()
            selected_categories = list(prompt_files_dict.keys())
            print(f"Using custom prompt files: {{len(selected_categories)}} categories")
        elif CONCEPTS_STR:
            # Parse concepts
            selected_concepts = [c.strip() for c in CONCEPTS_STR.split(',')]
            selected_categories = selected_concepts
        else:
            # Default to all concepts
            selected_concepts = ['english', 'german', 'french', 'italian', 'spanish', 'portuguese', 'chinese', 'japanese', 'indonesian']
            selected_categories = selected_concepts

        # ===== VALIDATE PROMPT FILES =====
        print(f"\\n===== PROMPT FILES VALIDATION =====")

        if prompt_files_dict:
            # Validate custom prompt files
            found_files = []
            missing_files = []
            for category, filepath in prompt_files_dict.items():
                if os.path.exists(filepath):
                    found_files.append(f"{{category}}: {{filepath}}")
                else:
                    missing_files.append(f"{{category}}: {{filepath}}")
            
            print(f"Custom prompt files: {{len(prompt_files_dict)}} specified")
            print(f"✓ Found {{len(found_files)}}/{{len(prompt_files_dict)}} files")
            for f in found_files:
                print(f"  ✓ {{f}}")
            if missing_files:
                print(f"⚠️  Missing files:")
                for f in missing_files:
                    print(f"  ✗ {{f}}")
        else:
            # Validate standard language files
            if not PROMPTS_DIR:
                raise ValueError("prompts_dir must be provided when not using custom prompt_files")
            if not os.path.exists(PROMPTS_DIR):
                raise FileNotFoundError(f"Prompts directory not found: {{PROMPTS_DIR}}")
            
            expected_files = [f'prompts_{{c}}.txt' for c in selected_concepts]
            found_files = []
            missing_files = []
            for fname in expected_files:
                fpath = os.path.join(PROMPTS_DIR, fname)
                if os.path.exists(fpath):
                    found_files.append(fname)
                else:
                    missing_files.append(fname)
            
            print(f"Prompts directory: {{PROMPTS_DIR}}")
            print(f"Selected concepts: {{', '.join(selected_concepts)}}")
            print(f"✓ Found {{len(found_files)}}/{{len(expected_files)}} expected files")
            if missing_files:
                print(f"⚠️  Missing files: {{', '.join(missing_files)}}")

        # ===== IMPORT REQUIRED MODULES =====
        print(f"\\n===== IMPORTING MODULES =====")

        # Core Python modules
        import re
        from typing import Dict, List, Tuple, Optional

        # Scientific computing
        try:
            import numpy as np
            print("✓ numpy")
        except Exception as e:
            print(f"✗ numpy: {{e}}")
            raise

        try:
            import pandas as pd
            print("✓ pandas")
        except Exception as e:
            print(f"✗ pandas: {{e}}")
            raise

        # Visualization
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            import seaborn as sns
            print("✓ matplotlib, seaborn")
        except Exception as e:
            print(f"✗ matplotlib/seaborn: {{e}}")
            raise

        # PyTorch
            try:
                import torch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"✓ torch (device: {{device}})")
        except Exception as e:
            print(f"✗ torch: {{e}}")
            raise

        # Transformers
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print("✓ transformers")
        except Exception as e:
            print(f"✗ transformers: {{e}}")
            raise

        # Single-cell analysis
            try:
                import scanpy as sc
                import anndata as ad
            sc.settings.verbosity = 1
            sc.settings.set_figure_params(dpi=300, facecolor='white')
            print("✓ scanpy, anndata")
        except Exception as e:
            print(f"✗ scanpy/anndata: {{e}}")
            raise

        # Scikit-learn
        try:
            from sklearn.preprocessing import StandardScaler
            print("✓ scikit-learn")
        except Exception as e:
            print(f"✗ scikit-learn: {{e}}")
            raise

        # Progress bars
            try:
                from tqdm import tqdm
            print("✓ tqdm")
        except Exception as e:
            print(f"⚠️  tqdm: {{e}} (optional)")

        # SAE (optional but recommended)
            try:
                from sae_lens import SAE
            print("✓ sae_lens")
        except Exception as e:
            print(f"⚠️  sae_lens: {{e}} (required for SAE analysis)")

        # ===== VERIFY PIPELINE MODULE =====
        print(f"\\n===== VERIFYING PIPELINE MODULE =====")
        try:
            from workspace.Agents.FeatureFinder.modules.featurefinder import CompleteNeuralAtlasPipeline
            print("✓ CompleteNeuralAtlasPipeline imported successfully")
        except Exception as e:
            print(f"✗ Failed to import CompleteNeuralAtlasPipeline: {{e}}")
            raise

        # ===== EXPORT ENVIRONMENT VARIABLES =====
        os.environ["WORKSPACE_ROOT"] = WORKSPACE_ROOT
        if PROMPTS_DIR:
            os.environ["PROMPTS_DIR"] = PROMPTS_DIR
        if prompt_files_dict:
            # Serialize prompt_files dict as JSON
            import json
            os.environ["PROMPT_FILES"] = json.dumps(prompt_files_dict)
        elif selected_concepts:
            os.environ["CONCEPTS"] = ','.join(selected_concepts)

        print(f"\\n===== ENVIRONMENT READY =====")
        print(f"Workspace:  {{WORKSPACE_ROOT}}")
        if prompt_files_dict:
            print(f"Mode:       Custom prompt files")
            print(f"Categories: {{', '.join(selected_categories)}} ({{len(selected_categories)}} total)")
        elif PROMPTS_DIR:
            print(f"Prompts:    {{PROMPTS_DIR}}")
            print(f"Concepts:   {{', '.join(selected_categories)}} ({{len(selected_categories)}} total)")
        print(f"\\nThe pipeline will create its own output directories based on model configuration.")
        print(f"You can now call run_neural_atlas_pipeline with your desired parameters.\\n")
            """
    return dedent(code)

@tool
def run_neural_atlas_pipeline(
    model_key: str = "gemma_2b",
    prompts_dir: str = "",
    results_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/results_gemma2")),
    sae_layer_idx: int = 0,
    max_prompts_per_category: int = 500,
    token_statistic: str = "all_tokens",
    last_k_tokens: int = 666,
    top_p_trim: float = 0.5,
    run_permutation_test: bool = False,
    n_permutations: int = 50,
    concepts: str = "",
    prompt_files: str = "",
) -> str:
    """
    Run the Complete Neural Atlas Pipeline with SAE encoding.
    
    This tool works in multiple modes:
    1. Standard concepts: Use `concepts` parameter or CONCEPTS env var
    2. Custom prompt files: Use `prompt_files` parameter or PROMPT_FILES env var
    3. With environment_setup_tool: Leave parameters empty to use environment variables
    
    Parameters:
        - model_key: Model identifier (default: "gemma_2b")
        - prompts_dir: Path to prompts folder (or empty to use env var PROMPTS_DIR)
        - results_dir: Path where results will be saved (optional, pipeline creates its own if empty)
        - sae_layer_idx: SAE layer index to analyze (0-25 for Gemma 2B)
        - max_prompts_per_category: Maximum number of prompts per category
        - token_statistic: Token aggregation method ("all_tokens", "mean_last_k", "max_last_k", "top_p_trimmed_mean")
        - last_k_tokens: Number of last tokens to consider for aggregation
        - top_p_trim: Fraction for top-p trimmed mean
        - run_permutation_test: Whether to run permutation testing (slow)
        - n_permutations: Number of permutations for testing
        - concepts: Comma-separated list of concepts to analyze (e.g., "english,chinese,french")
                    Leave empty to use CONCEPTS env var or analyze all available concepts.
                    Available: english, german, french, italian, spanish, portuguese, chinese, japanese, indonesian
        - prompt_files: Custom prompt files specification (takes precedence over concepts)
                       Format: "category1:/path/to/file1.txt,category2:/path/to/file2.txt"
                       Example: "english:/path/custom_en.txt,chinese:/path/custom_zh.txt"
    
    Note: The pipeline creates its own output directories based on model configuration.
    If results_dir is provided, it will be used; otherwise, pipeline creates model-specific directories.
    
    Please use python_repl_tool to execute this code directly. Do not modify this code arbitrarily.
    """
    code = f"""
        import sys
        import os

        # Add workspace to path for imports (use env var if available)
        workspace_path = os.environ.get("WORKSPACE_ROOT")
        if not workspace_path:
            workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if workspace_path not in sys.path:
            sys.path.insert(0, workspace_path)

        # Import the pipeline class
        from workspace.Agents.FeatureFinder.modules.featurefinder import CompleteNeuralAtlasPipeline

        # Configuration - use environment variables as fallback
        model_key = {model_key!r}
        prompts_dir = {prompts_dir!r} or os.environ.get("PROMPTS_DIR", "")
        results_dir = {results_dir!r}  # Empty means pipeline will create its own
        sae_layer_idx = {sae_layer_idx}
        max_prompts_per_category = {max_prompts_per_category}
        token_statistic = {token_statistic!r}
        last_k_tokens = {last_k_tokens}
        top_p_trim = {top_p_trim}
        run_permutation_test = {run_permutation_test}
        n_permutations = {n_permutations}

        # Parse prompt_files or concepts parameter
        prompt_files_str = {prompt_files!r} or os.environ.get("PROMPT_FILES", "")
        concepts_str = {concepts!r} or os.environ.get("CONCEPTS", "")

        prompt_files_dict = None
        concepts_list = None

        if prompt_files_str:
            # Parse custom prompt files
            import json
            try:
                # Try JSON first (from environment)
                prompt_files_dict = json.loads(prompt_files_str)
            except json.JSONDecodeError:
                # Parse as "category:/path,category:/path" format
                prompt_files_dict = {{}}
                for pair in prompt_files_str.split(','):
                    if ':' in pair:
                        category, filepath = pair.split(':', 1)
                        prompt_files_dict[category.strip()] = filepath.strip()
        elif concepts_str:
            # Parse concepts
            concepts_list = [c.strip() for c in concepts_str.split(',')]
        else:
            # Will use all concepts (None = default)
            pass

        # Validate prompts_dir (only if not using custom prompt files)
        if not prompt_files_dict:
            if not prompts_dir:
                raise ValueError("prompts_dir must be provided either as parameter or via PROMPTS_DIR environment variable")
            if not os.path.exists(prompts_dir):
                raise FileNotFoundError(f"Prompts directory not found: {{prompts_dir}}")

        print("=" * 80)
        print(f"NEURAL ATLAS PIPELINE - {{model_key.upper()}}")
        print("=" * 80)
        print(f"SAE Layer:         {{sae_layer_idx}}")
        print(f"Prompts/category:  {{max_prompts_per_category}}")
        print(f"Token statistic:   {{token_statistic}}")
        if prompt_files_dict:
            print(f"Mode:              Custom prompt files")
            print(f"Categories:        {{', '.join(prompt_files_dict.keys())}} ({{len(prompt_files_dict)}} total)")
        elif concepts_list:
            print(f"Prompts dir:       {{prompts_dir}}")
            print(f"Concepts:          {{', '.join(concepts_list)}} ({{len(concepts_list)}} selected)")
        else:
            print(f"Prompts dir:       {{prompts_dir}}")
            print(f"Concepts:          All available (9 concepts)")
        if run_permutation_test:
            print(f"Permutation test:  {{n_permutations}} permutations")

        # Create pipeline instance
        pipeline = CompleteNeuralAtlasPipeline(
            model_key=model_key,
            max_prompts_per_category=max_prompts_per_category,
            token_statistic=token_statistic,
            last_k_tokens=last_k_tokens,
            top_p_trim=top_p_trim,
            run_permutation_test=run_permutation_test,
            n_permutations=n_permutations,
            sae_layer_idx=sae_layer_idx,
            concepts=concepts_list,
            prompt_files=prompt_files_dict,
        )

        # Determine results directory
        if not results_dir:
            # Pipeline creates its own directories based on model config
            results_dir = pipeline.output_dir
            print(f"Using pipeline output dir: {{results_dir}}")
        else:
            print(f"Using specified results dir: {{results_dir}}")

        # Run complete pipeline
        results, adata, marker_neurons = pipeline.run_complete_pipeline(
            prompts_dir=prompts_dir,
            results_dir=results_dir,
        )

        print("\\n" + "=" * 80)
        print(f"PIPELINE COMPLETE FOR {{model_key.upper()}}!")
        print("=" * 80)
        print(f"Results directory:  {{results_dir}}")
        print(f"Processed data:     {{pipeline.processed_data_dir}}")
        print(f"Total observations: {{results['step1_data_extraction']['total_observations']:,}}")
        print(f"Total features:     {{results['step1_data_extraction']['total_features']:,}}")
        if 'step3_marker_neurons' in results:
            print(f"\\nMarker features by category:")
            for cat, count in results['step3_marker_neurons']['marker_features_found'].items():
                print(f"  {{cat:12s}}: {{count:4d}} features")
        print("=" * 80)
        """
    return dedent(code)