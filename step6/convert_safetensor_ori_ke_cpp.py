import subprocess
import sys
from pathlib import Path


MERGED_MODEL_PATH = Path(
    "/Users/fajri/Projects/testmodel/step5/merged_model"
)
LLAMA_CPP_PATH = Path("/Users/fajri/Projects/testmodel/step6/llama.cpp")
VENV_PYTHON = Path(
    "/Users/fajri/Projects/testmodel/step6/"
    ".venv-llama-cpp/bin/python"
)
GGUF_PATH = Path(
    "/Users/fajri/Projects/testmodel/step6/"
    "qwen3-alfajri-f32.gguf"
)


def check_files():
    required_files = [
        MERGED_MODEL_PATH / "config.json",
        MERGED_MODEL_PATH / "model.safetensors",
        MERGED_MODEL_PATH / "tokenizer.json",
        LLAMA_CPP_PATH / "convert_hf_to_gguf.py",
        VENV_PYTHON,
    ]

    missing_files = [
        path for path in required_files
        if not path.exists()
    ]

    if missing_files:
        details = "\n".join(
            f"- {path}" for path in missing_files
        )

        raise RuntimeError(
            "File yang diperlukan belum tersedia:\n"
            f"{details}\n\n"
            "Jalankan setup_llama_cpp.py terlebih dahulu."
        )


def main():
    check_files()

    if GGUF_PATH.exists():
        answer = input(
            "GGUF sudah tersedia. Buat ulang? [y/N]: "
        ).strip().lower()

        if answer not in {"y", "yes"}:
            print(f"Menggunakan file yang sudah ada: {GGUF_PATH}")
            return

    converter_path = LLAMA_CPP_PATH / "convert_hf_to_gguf.py"

    command = [
        VENV_PYTHON,
        converter_path,
        MERGED_MODEL_PATH,
        "--outfile",
        GGUF_PATH,
        "--outtype",
        "f32",
    ]

    print("\nMengonversi Safetensors menjadi GGUF F32...")
    print(f"> {' '.join(str(item) for item in command)}")

    subprocess.run(
        [str(item) for item in command],
        check=True,
    )

    size_gb = GGUF_PATH.stat().st_size / (1024**3)

    print("\nKonversi selesai.")
    print(f"File  : {GGUF_PATH}")
    print(f"Ukuran: {size_gb:.2f} GB")
    print("Format: F32 tanpa quantization Q4/Q8")
    print(
        "\nLanjutkan dengan:\n"
        "python3 /Users/fajri/Projects/testmodel/step6/"
        "deploy_ollama_versi_llama_cpp.py"
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        raise SystemExit(1)
