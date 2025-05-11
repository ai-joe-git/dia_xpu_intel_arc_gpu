@echo off
echo Setting up Dia XPU for Intel Arc GPU...

REM Clone the repository if it doesn't exist
if not exist dia_xpu_intel_arc_gpu (
    echo Cloning repository...
    git clone https://github.com/ai-joe-git/dia_xpu_intel_arc_gpu.git
    cd dia_xpu_intel_arc_gpu
) else (
    echo Repository already exists, using existing folder
    cd dia_xpu_intel_arc_gpu
)

REM Run the setup script if venv doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python setup_xpu.py
) else (
    echo Virtual environment already exists
)

REM Activate the virtual environment and run the app
echo Activating virtual environment and starting Gradio UI...
call venv\Scripts\activate.bat
python app.py --device xpu

pause
