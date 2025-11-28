### combine hist per type (ATTN, MlP, Norm, Block)

import re

def categorize_layer(layername: str) -> str:
    lname = layername.lower()

    # --- Skip sublayers (dropout, norm, activation, etc.) ---
    skip_patterns = [
        "drop", "dropout", "drop_path", "attn_drop", "proj_drop",
        "ln", "layernorm", "norm", "q_norm", "k_norm",
        "act", "activation", "activation_fn",
        "softmax", "resid_dropout"
    ]
    if any(x in lname for x in skip_patterns):
        return None

    # --- Skip attention *sublayers* ---
    if any(x in lname for x in [".qkv", ".proj", ".q_proj", ".k_proj", ".v_proj",
                                ".out_proj", ".c_proj", ".c_attn"]):
        return None

    # --- Attention modules ---
    if re.search(r'(attn|attention)$', lname):
        return "attention"

    # ------------------------------------------------------------------
    # MLP detection (updated!)
    # ------------------------------------------------------------------

    # 1) Pure MLP module: ends with ".mlp"
    if re.search(r'\bmlp$', lname):
        return "mlp"

    # 2) Ends with ".fc1" or ".fc2", but ONLY if "mlp" is NOT in the path
    #    Avoid matching ViT-style "blocks.X.mlp.fc1"
    if re.search(r'\.(fc1|fc2)$', lname) and "mlp" not in lname:
        return "mlp"

    # --- Skip standalone mlp sublayers ---
    if any(x in lname for x in [".c_fc", ".dense"]):
        return None

    # --- Block modules ---
    if re.search(r'\b(transformer\.h\.\d+|blocks\.\d+|layers\.\d+|encoder\.layers\.\d+|decoder\.layers\.\d+)$',
                 lname):
        return "block"

    # --- Fallback: standalone fc/dense (but not mlp sublayers) ---
    if re.search(r'\b(fc|dense|head\.fc|classification_head\.dense)$', lname):
        return "mlp"

    return None



def get_layer_color(layer_type):
    """
    Determine color based on layer type.
    Returns: (bar_color, min_marker_color, max_marker_color, edge_color)
    """
    if layer_type == "mlp":
        return '#00b050', '#006030', '#90EE90', '#004020'  # Green shades
    elif layer_type == "block":
        return '#4169E1', '#00008B', '#87CEEB', '#000080'  # Blue shades
    elif layer_type == "attention":
        return '#FF8C00', '#CC7000', '#FFB84D', '#995300'  # Orange shades
    # elif layer_type == "lnorm": --- IGNORE ---
    #     return '#808080', '#404040', '#C0C0C0', '#202020'  # Gray shades
    else:
        return '#D3D3D3', '#A9A9A9', '#E8E8E8', '#696969'  # Light gray for None/other
