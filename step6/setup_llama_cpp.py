import shutil
import subprocess
import sys
from pathlib import Path


LLAMA_CPP_PATH = Path("/Users/fajri/Projects/step6/llama.cpp")
VENV_PATH = Path(
    "/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp"
)
VENV_PYTHON = VENV_PATH / "bin" / "python"


def run(command):
    command = [str(item) for item in command]

    print(f"\n> {' '.join(command)}")
    subprocess.run(command, check=True)


def main():
    if shutil.which("git") is None:
        raise RuntimeError("Command git tidak ditemukan.")

    converter_path = LLAMA_CPP_PATH / "convert_hf_to_gguf.py"

    if not converter_path.exists():
        print("Mengunduh llama.cpp...")

        run(
            [
                "git",
                "clone",
                "https://github.com/ggml-org/llama.cpp.git",
                LLAMA_CPP_PATH,
            ]
        )
    else:
        print(f"llama.cpp sudah tersedia: {LLAMA_CPP_PATH}")

    if not VENV_PYTHON.exists():
        print("Membuat virtual environment converter...")

        run(
            [
                sys.executable,
                "-m",
                "venv",
                VENV_PATH,
            ]
        )

    requirements_path = LLAMA_CPP_PATH / "requirements.txt"

    if not requirements_path.exists():
        raise RuntimeError(
            f"File dependency tidak ditemukan: {requirements_path}"
        )

    print("Menginstal dependency converter...")

    run(
        [
            VENV_PYTHON,
            "-m",
            "pip",
            "install",
            "-r",
            requirements_path,
        ]
    )

    print("\nSetup llama.cpp selesai.")
    print(
        "Lanjutkan dengan:\n"
        "python3.11 /Users/fajri/Projects/testmodel/step6/"
        "convert_safetensor_ke_gguf.py"
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"\nError: {error}", file=sys.stderr)
        raise SystemExit(1)
