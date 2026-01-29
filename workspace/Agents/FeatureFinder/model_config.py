#!/usr/bin/env python3
"""
Model Configuration for Gemma Models
Only includes checkpoints that actually exist on Hugging Face

SAE results path (outside agentic-work): set GEMMA_SAE_RESULTS_BASE to the directory
that contains results_multitoken/ (e.g. path to zzz.gemma2SAE or equivalent).
If unset, falls back to relative path from workspace: ../zzz.gemma2SAE (when run from workspace).
"""

import os

# Gemma Model Configurations - Only existing checkpoints
GEMMA_MODELS = {
    "gemma_2b": {
        "name": "google/gemma-2-2b",
        "training_stage": "pretrained",
        "description": "Gemma 2B base model",
        "model_id": 0,
        "n_layers": 26,  # Gemma 2 2B has 26 layers
        "hidden_size": 2304,  # Gemma 2 2B has 2304 hidden units
        "vocab_size": 256000,  # Gemma 2 2B vocabulary size
        "max_position_embeddings": 8192  # Gemma 2 2B context length
    }
}

def get_model_config(model_key):
    """Get configuration for a specific model."""
    if model_key not in GEMMA_MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(GEMMA_MODELS.keys())}")
    return GEMMA_MODELS[model_key]

def get_model_revision(model_key):
    """Get the revision (checkpoint) for a specific model if it exists."""
    config = get_model_config(model_key)
    return config.get('revision', None)

def get_model_id(model_key):
    """Get the numeric model ID for a specific model."""
    config = get_model_config(model_key)
    return config.get('model_id', None)

def get_output_dir(model_key):
    """Get the output directory for a specific model."""
    model_config = get_model_config(model_key)
    model_name = model_config["name"].replace("/", "_").replace("-", "_")
    
    # For early training checkpoints, include the revision in the directory name
    if 'revision' in model_config:
        revision = model_config['revision'].replace("-", "_").replace("_", "_")
        return f"./results/{model_name}_{revision}"
    
    return f"./results/{model_name}"

def _get_sae_results_base():
    """Base path for SAE multitoken results (outside agentic-work). Prefer env, else relative to cwd."""
    base = os.environ.get("GEMMA_SAE_RESULTS_BASE") or os.environ.get("SAE_RESULTS_BASE")
    if base:
        return base.rstrip("/")
    # Fallback: relative to current working directory (e.g. run from workspace -> ../zzz.gemma2SAE)
    cwd = os.path.abspath(os.getcwd())
    rel = os.path.join(cwd, "..", "zzz.gemma2SAE")
    if os.path.isdir(rel):
        return os.path.normpath(rel)
    return os.path.normpath(rel)  # still return it; pipeline will fail with clear error if missing


def get_output_dir_multitoken(model_key):
    """Get the output directory for a specific model."""
    model_config = get_model_config(model_key)
    model_name = model_config["name"].replace("/", "_").replace("-", "_")
    base = _get_sae_results_base()
    # For early training checkpoints, include the revision in the directory name
    if 'revision' in model_config:
        revision = model_config['revision'].replace("-", "_").replace("_", "_")
        return os.path.join(base, "results_multitoken", f"{model_name}_{revision}")
    return os.path.join(base, "results_multitoken", model_name)


def get_output_dir_multitokenwEnglish(model_key):
    """Get the output directory for a specific model."""
    model_config = get_model_config(model_key)
    model_name = model_config["name"].replace("/", "_").replace("-", "_")
    base = _get_sae_results_base()
    # For early training checkpoints, include the revision in the directory name
    if 'revision' in model_config:
        revision = model_config['revision'].replace("-", "_").replace("_", "_")
        return os.path.join(base, "results_multitoken", f"{model_name}_{revision}")
    return os.path.join(base, "results_multitoken", model_name)


def get_all_model_keys():
    """Get list of all available model keys."""
    return list(GEMMA_MODELS.keys())

def print_model_summary():
    """Print a summary of all available models."""
    print("=" * 80)
    print("Gemma Models Available for Analysis (Only Existing Checkpoints)")
    print("=" * 80)
    
    print("\nFinal Models:")
    for key in GEMMA_MODELS.keys():
        config = GEMMA_MODELS[key]
        print(f"   {key}: {config['training_stage']}")
        print(f"      {config['description']}")
    
    print("\n" + "=" * 80)

