#!/usr/bin/env python3
import subprocess
import sys
import os

def create_venv_and_install():
    venv_dir = "./venv"
    if not os.path.exists(venv_dir):
        print(f"Creating virtual environment in {venv_dir}...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
    else:
        print(f"Virtual environment already exists in {venv_dir}.")

    # Activate venv and install dependencies
    pip_executable = os.path.join(venv_dir, "bin", "pip")
    if not os.path.exists(pip_executable):
        pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")  # Windows fallback

    print("Upgrading pip...")
    subprocess.check_call([pip_executable, "install", "--upgrade", "pip"])

    print("Installing PyTorch nightly with XPU support and Intel extension...")
    subprocess.check_call([
        pip_executable,
        "install",
        "--pre",
        "torch",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/nightly/"
    ])

    subprocess.check_call([
        pip_executable,
        "install",
        "intel-extension-for-pytorch"
    ])

    print("Installing other project dependencies...")
    subprocess.check_call([pip_executable, "install", "-e", "."])

    print("Setup complete. To activate the virtual environment, run:")
    print(f"source {venv_dir}/bin/activate  # On Linux/macOS")
    print(f"{venv_dir}\\Scripts\\activate.bat  # On Windows")


if __name__ == '__main__':
    create_venv_and_install()
