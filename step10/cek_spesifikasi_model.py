"""Tampilkan spesifikasi model hasil continue training Step 8.

Script hanya membaca config dan ukuran file. Model tidak dimuat ke RAM dan
tidak melakukan inference.
"""

import json
from pathlib import Path


STEP_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = (
    (
        "MODEL HASIL CONTINUE TRAINING STEP 8",
        STEP_DIR.parent / "step8" / "merged_model",
    ),
    (
        "MODEL HASIL MERGE STEP 5",
        STEP_DIR.parent / "step5" / "merged_model",
    ),
    (
        "BASE MODEL",
        Path("/Users/fajri/Projects/mytestmodel"),
    ),
)


def format_count(value):
    return f"{value:,}".replace(",", ".")


def format_size(size_bytes):
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def select_model():
    """Pilih model terbaru yang sudah tersedia di project."""
    for label, path in MODEL_CANDIDATES:
        if (path / "config.json").is_file():
            return label, path

    checked_paths = "\n".join(f"- {path}" for _, path in MODEL_CANDIDATES)
    raise FileNotFoundError(
        "Tidak ada config model yang ditemukan. Folder yang diperiksa:\n"
        f"{checked_paths}"
    )


def calculate_qwen_parameters(config):
    """Hitung parameter Qwen3 berdasarkan bentuk matrix pada config."""
    vocabulary = config["vocab_size"]
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    layers = config["num_hidden_layers"]
    attention_heads = config["num_attention_heads"]
    key_value_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", hidden_size // attention_heads)

    query_size = attention_heads * head_dim
    key_value_size = key_value_heads * head_dim

    embedding = vocabulary * hidden_size
    q_proj = hidden_size * query_size
    k_proj = hidden_size * key_value_size
    v_proj = hidden_size * key_value_size
    o_proj = query_size * hidden_size
    attention_per_layer = q_proj + k_proj + v_proj + o_proj

    gate_proj = hidden_size * intermediate_size
    up_proj = hidden_size * intermediate_size
    down_proj = intermediate_size * hidden_size
    mlp_per_layer = gate_proj + up_proj + down_proj

    normalization_per_layer = 2 * hidden_size
    final_normalization = hidden_size
    layer_total = attention_per_layer + mlp_per_layer + normalization_per_layer
    lm_head = 0 if config.get("tie_word_embeddings") else vocabulary * hidden_size
    total = embedding + (layer_total * layers) + final_normalization + lm_head

    return {
        "embedding": embedding,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "o_proj": o_proj,
        "attention_per_layer": attention_per_layer,
        "mlp_per_layer": mlp_per_layer,
        "normalization_per_layer": normalization_per_layer,
        "layer_total": layer_total,
        "all_layers": layer_total * layers,
        "final_normalization": final_normalization,
        "lm_head": lm_head,
        "total": total,
    }


def main():
    model_label, model_path = select_model()
    config_path = model_path / "config.json"
    config = read_json(config_path)
    params = calculate_qwen_parameters(config)
    weight_files = sorted(model_path.glob("*.safetensors"))
    total_weight_size = sum(path.stat().st_size for path in weight_files)

    print(f"SPESIFIKASI {model_label}")
    print("=" * 62)
    print(f"Folder               : {model_path}")
    print(f"Architecture         : {', '.join(config.get('architectures', []))}")
    print(f"Model type           : {config.get('model_type')}")
    print(f"Transformer layer    : {config.get('num_hidden_layers')}")
    print(f"Hidden size          : {format_count(config.get('hidden_size'))}")
    print(f"Intermediate MLP     : {format_count(config.get('intermediate_size'))}")
    print(f"Attention heads      : {config.get('num_attention_heads')}")
    print(f"Key/value heads      : {config.get('num_key_value_heads')}")
    print(f"Head dimension       : {config.get('head_dim')}")
    print(f"Vocabulary           : {format_count(config.get('vocab_size'))} token")
    print(f"Context model        : {format_count(config.get('max_position_embeddings'))} token")
    print(f"Embedding tied lmhead: {config.get('tie_word_embeddings')}")
    print(f"Data type config     : {config.get('torch_dtype', config.get('dtype'))}")

    print("\nPERHITUNGAN PARAMETER")
    print("=" * 62)
    print(f"Embedding table      : {format_count(params['embedding'])}")
    print(f"q_proj / layer       : {format_count(params['q_proj'])}")
    print(f"k_proj / layer       : {format_count(params['k_proj'])}")
    print(f"v_proj / layer       : {format_count(params['v_proj'])}")
    print(f"o_proj / layer       : {format_count(params['o_proj'])}")
    print(f"Attention / layer    : {format_count(params['attention_per_layer'])}")
    print(f"MLP / layer          : {format_count(params['mlp_per_layer'])}")
    print(f"Norm / layer         : {format_count(params['normalization_per_layer'])}")
    print(f"Total / layer        : {format_count(params['layer_total'])}")
    print(f"Total semua layer    : {format_count(params['all_layers'])}")
    print(f"Final norm           : {format_count(params['final_normalization'])}")
    print(f"lm_head tambahan     : {format_count(params['lm_head'])}")
    print(f"TOTAL PARAMETER      : {format_count(params['total'])}")
    print(f"≈ ukuran model       : {params['total'] / 1_000_000_000:.3f}B")

    print("\nFILE BOBOT")
    print("=" * 62)
    print(f"Jumlah safetensors   : {len(weight_files)}")
    print(f"Ukuran file bobot    : {format_size(total_weight_size)}")
    print("\n0.6B adalah jumlah parameter model. Context window, KV cache,")
    print("dan tensor inference tidak termasuk dalam total parameter tersebut.")


if __name__ == "__main__":
    main()
