"""
Neuronpedia API interactions for fetching SAE feature activation data.
Adapted from autonomous_feature_discovery_agent.
"""

import requests
from typing import Dict, List, Optional, Any


def get_top_activations(layer: str, feature_index: int, 
                       model: str = "gemma-2-2b") -> Optional[Dict]:
    """
    Fetch feature activation data from Neuronpedia API.
    
    Args:
        layer: Layer specification (e.g., "0-gemmascope-mlp-16k")
        feature_index: Feature index
        model: Model name (default: "gemma-2-2b")
        
    Returns:
        Dictionary with activation data or None if fetch fails
    """
    url = f"https://www.neuronpedia.org/api/feature/{model}/{layer}/{feature_index}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Error fetching from Neuronpedia: {e}")
        return None


def parse_activations(feature_data: Dict) -> List[Dict]:
    """
    Parse activation data from Neuronpedia format.
    
    Args:
        feature_data: Raw data from Neuronpedia API
        
    Returns:
        List of parsed activation examples
    """
    if not feature_data or 'activations' not in feature_data:
        return []
    
    parsed = []
    for act in feature_data.get('activations', []):
        tokens = act.get('tokens', [])
        values = act.get('values', [])
        max_value = act.get('maxValue', 0.0)
        max_token_idx = act.get('maxValueTokenIndex', -1)
        
        max_token = tokens[max_token_idx] if 0 <= max_token_idx < len(tokens) else ''
        text = ''.join(tokens).replace('\u2581', ' ')
        
        parsed.append({
            'text': text,
            'tokens': tokens,
            'activation_values': values,
            'max_activation_value': max_value,
            'max_activation_token': max_token,
            'max_activation_index': max_token_idx
        })
    
    return parsed


def extract_logit_info(feature_data: Dict) -> Dict[str, Any]:
    """
    Extract logit information from feature data.
    
    Args:
        feature_data: Raw data from Neuronpedia API
        
    Returns:
        Dictionary with top positive and negative logits
    """
    logit_info = {
        'top_positive_logits': [],
        'top_negative_logits': [],
        'logit_distribution': None
    }
    
    if 'pos_str' in feature_data and 'pos_values' in feature_data:
        pos_tokens = feature_data['pos_str']
        pos_values = feature_data['pos_values']
        logit_info['top_positive_logits'] = [
            {'token': token.replace('\u2581', ' '), 'value': float(val)}
            for token, val in zip(pos_tokens[:10], pos_values[:10])
        ]
    
    if 'neg_str' in feature_data and 'neg_values' in feature_data:
        neg_tokens = feature_data['neg_str']
        neg_values = feature_data['neg_values']
        logit_info['top_negative_logits'] = [
            {'token': token.replace('\u2581', ' '), 'value': float(val)}
            for token, val in zip(neg_tokens[:10], neg_values[:10])
        ]
    
    return logit_info


def format_layer_for_neuronpedia(sae_layer: int, width: str = "16k") -> str:
    """
    Convert SAE layer number to Neuronpedia layer format.
    
    Args:
        sae_layer: Layer number (e.g., 0)
        width: SAE width (default: "16k")
        
    Returns:
        Formatted string (e.g., "0-gemmascope-mlp-16k")
    """
    return f"{sae_layer}-gemmascope-mlp-{width}"

