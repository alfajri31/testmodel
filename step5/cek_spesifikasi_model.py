import json
from pathlib import Path

from transformers import AutoModelForCausalLM


BASE_MODEL_PATH = Path("/Users/fajri/Projects/mytestmodel")
ADAPTER_PATH = Path("/Users/fajri/Projects/testmodel/step5/output")
MERGED_MODEL_PATH = Path(
    "/Users/fajri/Projects/testmodel/step5/merged_model"
)


def format_size(size_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024


def read_json(path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def print_neural_network_structure():
    print("\nSTRUKTUR NEURAL NETWORK BASE MODEL")
    print("=" * 60)
    print(f"Folder               : {BASE_MODEL_PATH}")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        local_files_only=True,
    )

    print()
    print(model)


def print_model_specification(name, model_path):
    config_path = model_path / "config.json"
    tokenizer_config_path = model_path / "tokenizer_config.json"
    weights = sorted(model_path.glob("*.safetensors"))

    print(f"\n{name}")
    print("=" * 60)
    print(f"Folder               : {model_path}")

    if not config_path.exists():
        print(f"Config               : tidak ditemukan ({config_path})")
        return

    config = read_json(config_path)
    tokenizer_config = (
        read_json(tokenizer_config_path)
        if tokenizer_config_path.exists()
        else {}
    )

    architecture = config.get("architectures") or []

    print(f"Architecture         : {', '.join(architecture)}")
    print(f"Model type           : {config.get('model_type')}")
    print(f"Hidden size          : {config.get('hidden_size')}")
    print(f"Intermediate size    : {config.get('intermediate_size')}")
    print(f"Jumlah layer         : {config.get('num_hidden_layers')}")
    print(f"Attention heads      : {config.get('num_attention_heads')}")
    print(f"Key/value heads      : {config.get('num_key_value_heads')}")
    print(f"Vocabulary token     : {config.get('vocab_size'):,}")
    print(
        "Context model       : "
        f"{config.get('max_position_embeddings'):,} token"
    )

    tokenizer_limit = tokenizer_config.get("model_max_length")
    if isinstance(tokenizer_limit, int):
        print(f"Limit tokenizer      : {tokenizer_limit:,} token")
    else:
        print(f"Limit tokenizer      : {tokenizer_limit}")

    print(f"Data type config     : {config.get('torch_dtype')}")

    if weights:
        total_size = sum(weight.stat().st_size for weight in weights)
        print(f"File bobot           : {len(weights)} file")
        print(f"Ukuran bobot         : {format_size(total_size)}")
    else:
        print("File bobot           : tidak ditemukan")


def print_adapter_specification():
    config_path = ADAPTER_PATH / "adapter_config.json"
    weights_path = ADAPTER_PATH / "adapter_model.safetensors"

    print("\nLORA ADAPTER")
    print("=" * 60)
    print(f"Folder               : {ADAPTER_PATH}")

    if not config_path.exists():
        print(f"Config               : tidak ditemukan ({config_path})")
        return

    config = read_json(config_path)

    print(f"Base model           : {config.get('base_model_name_or_path')}")
    print(f"LoRA rank            : {config.get('r')}")
    print(f"LoRA alpha           : {config.get('lora_alpha')}")
    print(f"LoRA dropout         : {config.get('lora_dropout')}")
    print(
        "Target modules      : "
        f"{', '.join(config.get('target_modules', []))}"
    )

    if weights_path.exists():
        print(
            "Ukuran adapter      : "
            f"{format_size(weights_path.stat().st_size)}"
        )
    else:
        print(f"Bobot adapter        : tidak ditemukan ({weights_path})")


print_neural_network_structure()
print_model_specification("BASE MODEL", BASE_MODEL_PATH)
print_adapter_specification()
print_model_specification("MERGED MODEL", MERGED_MODEL_PATH)

print("\nPENJELASAN TOKEN")
print("=" * 60)
print("Vocabulary token = jumlah jenis token di dalam kamus tokenizer.")
print("Context model    = jumlah token yang dapat diproses sekaligus.")
print("Gunakan context model sebagai batas aman jika nilainya berbeda.")
print("Token vocabulary tidak berkurang dan tidak dapat habis.")
