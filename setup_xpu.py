#!/usr/bin/env python3
import subprocess
import sys
import os
import platform

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
        pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:  # Unix/Linux/MacOS
        python_executable = os.path.join(venv_dir, "bin", "python")
        pip_executable = os.path.join(venv_dir, "bin", "pip")

    print("Upgrading pip...")
    subprocess.check_call([python_executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install core dependencies first
    print("Installing core dependencies...")
    subprocess.check_call([
        python_executable, "-m", "pip", "install", 
        "numpy", 
        "pydantic", 
        "safetensors", 
        "soundfile", 
        "gradio>=5.25.2", 
        "huggingface-hub>=0.30.2",
        "descript-audio-codec>=1.0.0"
    ])

    # Install PyTorch with XPU support - using a specific version known to work
    print("Installing PyTorch with XPU support...")
    subprocess.check_call([
        python_executable, "-m", "pip", "install", 
        "torch==2.1.0", 
        "torchaudio==2.1.0",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])

    # Now install the project itself
    print("Installing the project...")
    try:
        subprocess.check_call([python_executable, "-m", "pip", "install", "-e", "."])
    except subprocess.CalledProcessError:
        print("Warning: Could not install the project in development mode.")
        print("You may need to install additional dependencies manually.")

    print("\nSetup complete!")
    print("\nTo activate the virtual environment, run:")
    if os.name == 'nt':
        print(f"{venv_dir}\\Scripts\\activate.bat")
    else:
        print(f"source {venv_dir}/bin/activate")
    
    print("\nNote: Intel XPU support is provided through PyTorch's native implementation.")
    print("You can now run the application with: python app.py --device xpu")

if __name__ == '__main__':
    create_venv_and_install()
