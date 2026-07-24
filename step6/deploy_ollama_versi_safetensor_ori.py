import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


MODEL_NAME = "alfajri-qwen3"

MERGED_MODEL_PATH = Path(
    "/Users/fajri/Projects/testmodel/step5/merged_model"
)

STEP6_PATH = Path(
    "/Users/fajri/Projects/testmodel/step6"
)

MODELFILE_PATH = STEP6_PATH / "Modelfile"

OLLAMA_URL = "http://127.0.0.1:11434"


def check_merged_model():
    required_files = [
        MERGED_MODEL_PATH / "config.json",
        MERGED_MODEL_PATH / "model.safetensors",
        MERGED_MODEL_PATH / "tokenizer.json",
        MERGED_MODEL_PATH / "tokenizer_config.json",
    ]

    missing_files = [
        path for path in required_files
        if not path.exists()
    ]

    if missing_files:
        print("File merged model belum lengkap:")

        for path in missing_files:
            print(f"- {path}")

        raise SystemExit(1)


def check_ollama_server():
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_URL}/api/version",
            timeout=3,
        ) as response:
            data = json.loads(response.read())

        print(f"Ollama server aktif: {data.get('version')}")

    except (urllib.error.URLError, TimeoutError):
        print("Ollama server belum berjalan.")
        print("Jalankan pada terminal lain:")
        print()
        print("ollama serve")

        raise SystemExit(1)


def create_modelfile():
    STEP6_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = f"""FROM {MERGED_MODEL_PATH}

PARAMETER num_ctx 4096
PARAMETER temperature 0

SYSTEM \"\"\"Jawab pertanyaan dengan singkat dan jelas berdasarkan informasi yang telah dipelajari.\"\"\"
"""

    MODELFILE_PATH.write_text(
        content,
        encoding="utf-8",
    )

    print(f"Modelfile dibuat: {MODELFILE_PATH}")


def deploy_model():
    command = [
        "ollama",
        "create",
        MODEL_NAME,
        "--experimental",
        "--file",
        str(MODELFILE_PATH),
    ]

    print()
    print("Men-deploy model ke Ollama...")
    print(" ".join(command))
    print()

    subprocess.run(
        command,
        check=True,
    )


def show_models():
    print()
    print("Daftar model Ollama:")

    subprocess.run(
        ["ollama", "list"],
        check=True,
    )


def main():
    check_merged_model()
    check_ollama_server()
    create_modelfile()
    deploy_model()
    show_models()

    print()
    print("Deploy selesai.")
    print("Jalankan model dengan:")
    print()
    print(f"ollama run {MODEL_NAME}")


if __name__ == "__main__":
    try:
        main()

    except FileNotFoundError:
        print("Command ollama tidak ditemukan.")
        print("Pastikan Ollama sudah terpasang.")
        raise SystemExit(1)

    except subprocess.CalledProcessError as error:
        print()
        print(f"Deploy gagal dengan exit code {error.returncode}.")
        raise SystemExit(error.returncode)

    except KeyboardInterrupt:
        print("\nProses dibatalkan.")
        sys.exit(130)
