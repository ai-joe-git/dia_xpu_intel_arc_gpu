import os
import time
import tempfile
import argparse
import numpy as np
import torch
import gradio as gr
from dia.model import Dia

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Dia TTS Demo")
parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, xpu)")
parser.add_argument("--model", type=str, default="nari-labs/Dia-1.6B", help="Model name or path")
parser.add_argument("--dtype", type=str, default="float16", help="Compute dtype (float32, float16, bfloat16)")
args = parser.parse_args()

# Set device based on arguments or auto-detect
if args.device:
    device = torch.device(args.device)
else:
    if torch.xpu.is_available():
        device = torch.device("xpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

print(f"Using device: {device}")

# Load model
model = Dia.from_pretrained(args.model, compute_dtype=args.dtype, device=device)

# Define interface functions
def run_inference(
    text,
    temperature=1.0,
    top_p=0.9,
    cfg_scale=3.0,
    audio_prompt=None,
    use_compile=True,
    max_tokens=None,
):
    """Run inference with the model"""
    try:
        # Prepare generation parameters
        generation_params = {
            "temperature": temperature,
            "top_p": top_p,
            "cfg_scale": cfg_scale,
            "use_torch_compile": use_compile,
            "verbose": True,
        }
        
        if max_tokens:
            generation_params["max_tokens"] = max_tokens
            
        if audio_prompt:
            generation_params["audio_prompt"] = audio_prompt
        
        # Use CPU-only generation to avoid audio distortion issues
        start_time = time.time()
        output_audio_np = model.generate_cpu_only(text, **generation_params)
        generation_time = time.time() - start_time
        print(f"Generation finished in {generation_time:.2f} seconds.")
        
        # Process the output audio
        if output_audio_np is None:
            return None
        
        # Normalize and convert to the right format for Gradio
        max_val = np.max(np.abs(output_audio_np))
        if max_val > 0:
            output_audio_np = output_audio_np / max_val * 0.9
        
        # Resample if needed
        original_samples = len(output_audio_np)
        target_samples = int(original_samples * 1.064)  # Slight adjustment to fix playback speed
        resampled = np.interp(
            np.linspace(0, original_samples - 1, target_samples),
            np.arange(original_samples),
            output_audio_np
        )
        print(f"Resampled audio from {original_samples} to {target_samples} samples for {original_samples/target_samples:.2f}x speed.")
        
        # Convert to int16 for Gradio
        audio_int16 = (resampled * 32767).astype(np.int16)
        print(f"Audio conversion successful. Final shape: {resampled.shape}, Sample Rate: {44100}")
        print("Converted audio to int16 for Gradio output.")
        
        return (44100, audio_int16)
    
    except Exception as e:
        raise gr.Error(f"Inference failed: {e}")

def voice_clone_inference(
    text,
    audio_file,
    temperature=1.0,
    top_p=0.9,
    cfg_scale=3.0,
    use_compile=True,
    max_tokens=None,
):
    """Run voice cloning inference"""
    try:
        return run_inference(
            text=text,
            temperature=temperature,
            top_p=top_p,
            cfg_scale=cfg_scale,
            audio_prompt=audio_file,
            use_compile=use_compile,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise gr.Error(f"Voice cloning failed: {e}")

def save_audio_file(audio):
    """Save audio to a temporary file and return the path"""
    if audio is None:
        return None
    
    sr, data = audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        import soundfile as sf
        sf.write(f.name, data, sr)
        return f.name

# Define the Gradio interface
with gr.Blocks(title="Dia TTS Demo") as demo:
    gr.Markdown("# Dia TTS Demo")
    gr.Markdown(f"Running on device: {device}")
    
    with gr.Tab("Text to Speech"):
        with gr.Row():
            with gr.Column():
                text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="[S1] Enter your text here. [S2] You can use [S1] and [S2] to indicate different speakers.",
                    lines=5,
                )
                
                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                    )
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                    )
                    cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=7.0,
                        value=3.0,
                        step=0.5,
                    )
                
                with gr.Row():
                    use_compile = gr.Checkbox(
                        label="Use torch.compile",
                        value=True,
                    )
                    max_tokens = gr.Number(
                        label="Max Tokens (leave empty for default)",
                        value=None,
                    )
                
                submit_btn = gr.Button("Generate Audio")
            
            with gr.Column():
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="numpy",
                )
                download_btn = gr.Button("Download Audio")
    
    with gr.Tab("Voice Cloning"):
        with gr.Row():
            with gr.Column():
                clone_text_input = gr.Textbox(
                    label="Text Input",
                    placeholder="[S1] Enter your text here. [S2] You can use [S1] and [S2] to indicate different speakers.",
                    lines=5,
                )
                
                audio_prompt = gr.Audio(
                    label="Voice to Clone (Upload an audio file)",
                    type="filepath",
                )
                
                with gr.Row():
                    clone_temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                    )
                    clone_top_p = gr.Slider(
                        label="Top P",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                    )
                    clone_cfg_scale = gr.Slider(
                        label="CFG Scale",
                        minimum=1.0,
                        maximum=7.0,
                        value=3.0,
                        step=0.5,
                    )
                
                with gr.Row():
                    clone_use_compile = gr.Checkbox(
                        label="Use torch.compile",
                        value=True,
                    )
                    clone_max_tokens = gr.Number(
                        label="Max Tokens (leave empty for default)",
                        value=None,
                    )
                
                clone_submit_btn = gr.Button("Generate Audio with Voice Cloning")
            
            with gr.Column():
                clone_audio_output = gr.Audio(
                    label="Generated Audio with Cloned Voice",
                    type="numpy",
                )
                clone_download_btn = gr.Button("Download Audio")
    
    # Set up event handlers
    submit_btn.click(
        fn=run_inference,
        inputs=[
            text_input,
            temperature,
            top_p,
            cfg_scale,
            None,  # No audio prompt
            use_compile,
            max_tokens,
        ],
        outputs=audio_output,
    )
    
    clone_submit_btn.click(
        fn=voice_clone_inference,
        inputs=[
            clone_text_input,
            audio_prompt,
            clone_temperature,
            clone_top_p,
            clone_cfg_scale,
            clone_use_compile,
            clone_max_tokens,
        ],
        outputs=clone_audio_output,
    )
    
    # Download buttons
    download_btn.click(
        fn=save_audio_file,
        inputs=audio_output,
        outputs=gr.File(label="Download"),
    )
    
    clone_download_btn.click(
        fn=save_audio_file,
        inputs=clone_audio_output,
        outputs=gr.File(label="Download"),
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch()
