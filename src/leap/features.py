"""Last valid prompt-token selection; no generation or future-token features."""
import numpy as np


def last_prompt_features(hidden, attention_mask):
    """Select the last nonpadding token, supporting either padding direction."""
    x = np.asarray(hidden)
    mask = np.asarray(attention_mask)
    if x.ndim != 3 or mask.shape != x.shape[:2] or not np.isfinite(x).all():
        raise ValueError('Expected finite [batch,tokens,features] and matching mask')
    if not np.isin(mask, [0, 1]).all() or not mask.any(axis=1).all():
        raise ValueError('Every prompt requires at least one nonpadding token')
    positions = np.where(mask.astype(bool), np.arange(x.shape[1]), -1).max(axis=1)
    return np.ascontiguousarray(x[np.arange(len(x)), positions], dtype=np.float32)


def select_decoder_layer(model, one_based_layer):
    """Resolve a Hugging Face decoder block explicitly; no silent layer fallback."""
    core = getattr(model, 'model', model)
    layers = getattr(core, 'layers', None)
    if layers is None or type(one_based_layer) is not int or not 1 <= one_based_layer <= len(layers):
        raise ValueError('Expected model.layers or model.model.layers and a valid one-based block')
    return layers[one_based_layer-1]

