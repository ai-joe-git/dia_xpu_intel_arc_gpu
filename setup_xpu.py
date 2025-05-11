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

    # Get Python executable path for the virtual environment
    if os.name == 'nt':  # Windows
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Unix/Linux/MacOS
        python_executable = os.path.join(venv_dir, "bin", "python")

    print("Upgrading pip...")
    # Use Python to run pip as a module (safer approach, especially on Windows)
    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])

    print("Installing PyTorch nightly with XPU support...")
    subprocess.check_call([
        python_executable, "-m", "pip", "install", "--pre", "torch", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/nightly/"
    ])

    print("Installing Intel Extension for PyTorch...")
    try:
        # Try to install the specific version from Intel's index
        subprocess.check_call([
            python_executable, "-m", "pip", "install", "intel-extension-for-pytorch==2.5.10+xpu",
            "--extra-index-url", "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
        ])
    except subprocess.CalledProcessError:
        print("Warning: Failed to install specific version of Intel Extension for PyTorch.")
        print("Trying alternative installation method...")
        try:
            # Try installing without version specification
            subprocess.check_call([
                python_executable, "-m", "pip", "install", "intel-extension-for-pytorch",
                "--extra-index-url", "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
            ])
        except subprocess.CalledProcessError:
            print("Warning: Intel Extension for PyTorch installation failed.")
            print("The model will still work but may not be optimized for Intel GPUs.")

    print("Installing required dependencies...")
    subprocess.check_call([
        python_executable, "-m", "pip", "install", "gradio", "huggingface-hub", "soundfile", 
        "numpy", "pydantic", "safetensors", "descript-audio-codec"
    ])

    print("Installing project in development mode...")
    try:
        subprocess.check_call([python_executable, "-m", "pip", "install", "-e", "."])
    except subprocess.CalledProcessError:
        print("Warning: Failed to install project in development mode.")
        print("Installing individual dependencies instead...")

    print("Setup complete. To activate the virtual environment, run:")
    print(f"source {venv_dir}/bin/activate  # On Linux/macOS")
    print(f"{venv_dir}\\Scripts\\activate.bat  # On Windows")


if __name__ == '__main__':
    create_venv_and_install()
