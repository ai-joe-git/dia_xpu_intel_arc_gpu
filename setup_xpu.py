#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil

def create_venv_and_install():
    venv_dir = "./venv"
    
    # If venv exists but has issues, remove it and create fresh
    if os.path.exists(venv_dir):
        print(f"Removing existing virtual environment in {venv_dir}...")
        try:
            shutil.rmtree(venv_dir)
        except Exception as e:
            print(f"Warning: Could not remove existing venv: {e}")
            print("Please manually delete the venv folder and try again.")
            return
    
    print(f"Creating fresh virtual environment in {venv_dir}...")
    subprocess.check_call([sys.executable, "-m", "venv", venv_dir])

    # Get Python executable path for the virtual environment
    if os.name == 'nt':  # Windows
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
    else:  # Unix/Linux/MacOS
        python_executable = os.path.join(venv_dir, "bin", "python")

    print("Upgrading pip...")
    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install NumPy 1.x first to avoid compatibility issues
    print("Installing NumPy 1.x...")
    subprocess.check_call([
        python_executable, "-m", "pip", "install", "numpy<2.0.0"
    ])

    # Install PyTorch 2.5.1 with XPU support (compatible with Intel Extension 2.5.10+xpu)
    print("Installing PyTorch 2.5.1 with XPU support...")
    subprocess.check_call([
        python_executable, "-m", "pip", "install", 
        "torch==2.5.1+xpu", 
        "torchaudio==2.5.1+xpu",
        "--index-url", "https://download.pytorch.org/whl/xpu"
    ])

    # Install Intel Extension for PyTorch 2.5.10+xpu
    print("Installing Intel Extension for PyTorch...")
    subprocess.check_call([
        python_executable, "-m", "pip", "install",
        "intel-extension-for-pytorch==2.5.10+xpu",
        "--extra-index-url", "https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
    ])

    # Install core dependencies
    print("Installing core dependencies...")
    subprocess.check_call([
        python_executable, "-m", "pip", "install", 
        "pydantic<3.0.0",
        "safetensors",
        "soundfile",
        "gradio>=5.25.2",
        "huggingface-hub>=0.30.2",
        "descript-audio-codec>=1.0.0"
    ])

    # Install triton based on platform
    if sys.platform == 'linux':
        subprocess.check_call([
            python_executable, "-m", "pip", "install", "triton==3.2.0"
        ])
    elif sys.platform == 'win32':
        subprocess.check_call([
            python_executable, "-m", "pip", "install", "triton-windows==3.2.0.post18"
        ])

    # Now install the project itself
    print("Installing the project...")
    try:
        subprocess.check_call([python_executable, "-m", "pip", "install", "-e", "."])
    except subprocess.CalledProcessError:
        print("Warning: Could not install the project in development mode.")
        print("Installing remaining dependencies directly...")
        subprocess.check_call([
            python_executable, "-m", "pip", "install",
            "transformers",
            "accelerate",
            "einops"
        ])

    print("\nSetup complete!")
    print("\nTo activate the virtual environment, run:")
    if os.name == 'nt':
        print(f"{venv_dir}\\Scripts\\activate.bat")
    else:
        print(f"source {venv_dir}/bin/activate")
    
    print("\nYou can now run the application with: python app.py --device xpu")

if __name__ == '__main__':
    create_venv_and_install()
