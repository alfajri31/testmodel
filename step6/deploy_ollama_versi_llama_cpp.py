import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


MODEL_NAME = "alfajri-qwen3-gguf-f32"
GGUF_PATH = Path(
    "/Users/fajri/Projects/testmodel/step6/"
    "qwen3-alfajri-f32.gguf"
)
MODELFILE_PATH = Path(
    "/Users/fajri/Projects/testmodel/step6/"
    "Modelfile.llama_cpp"
)
OLLAMA_URL = "http://127.0.0.1:11434"


def check_ollama_server():
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_URL}/api/version",
            timeout=3,
        ) as response:
            data = json.loads(response.read())

        print(f"Ollama server aktif: {data.get('version')}")

    except (urllib.error.URLError, TimeoutError):
        raise RuntimeError(
            "Ollama server belum berjalan. "
            "Buka aplikasi Ollama atau jalankan 'ollama serve'."
        )


def create_modelfile():
    content = f"""FROM {GGUF_PATH}

PARAMETER num_ctx 4096
PARAMETER temperature 0

SYSTEM \"\"\"Jawab pertanyaan dengan singkat dan jelas berdasarkan informasi yang telah dipelajari.\"\"\"
"""

    MODELFILE_PATH.write_text(
        content,
        encoding="utf-8",
    )

    print(f"Modelfile dibuat: {MODELFILE_PATH}")


def main():
    if shutil.which("ollama") is None:
        raise RuntimeError("Command ollama tidak ditemukan.")

    if not GGUF_PATH.exists():
        raise RuntimeError(
            f"File GGUF belum tersedia: {GGUF_PATH}\n"
            "Jalankan convert_safetensor_ke_gguf.py terlebih dahulu."
        )

    check_ollama_server()
    create_modelfile()

    command = [
        "ollama",
        "create",
        MODEL_NAME,
        "--file",
        MODELFILE_PATH,
    ]

    print(f"\n> {' '.join(str(item) for item in command)}")

    subprocess.run(
        [str(item) for item in command],
        check=True,
    )

    print("\nDeploy selesai.")
    print(f"Jalankan: ollama run {MODEL_NAME}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        raise SystemExit(1)
