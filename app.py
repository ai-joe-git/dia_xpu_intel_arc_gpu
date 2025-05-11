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
parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu, cuda, xpu)")
parser.add_argument("--model", type=str, default="nari-labs/Dia-1.6B", help="Model name or path")
parser.add_argument("--dtype", type=str, default="float32", help="Compute dtype (float32, float16, bfloat16)")
args = parser.parse_args()

# Force CPU usage regardless of what's available
device = torch.device("cpu")
print(f"Using device: {device}")

# Load model
model = Dia.from_pretrained(args.model, compute_dtype=args.dtype, device=device)

def run_inference(text, audio_prompt=None, temperature=1.0, top_p=0.9, cfg_scale=3.0, max_tokens=None):
    """Run inference with the model"""
    try:
        # Prepare generation parameters
        generation_params = {
            "temperature": temperature,
            "top_p": top_p,
            "cfg_scale": cfg_scale,
            "use_torch_compile": False,  # Explicitly disable torch.compile
            "verbose": True,
        }
        
        if max_tokens is not None and max_tokens > 0:
            generation_params["max_tokens"] = int(max_tokens)
            
        if audio_prompt is not None:
            generation_params["audio_prompt"] = audio_prompt
        
        # Use standard generation on CPU
        start_time = time.time()
        output_audio_np = model.generate(text, **generation_params)
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

# Create the Gradio interface - original style
with gr.Blocks() as demo:
    gr.Markdown("# Dia - Text to Speech Model (CPU Mode)")
    gr.Markdown("Enter text with speaker tags like [S1] and [S2]. You can also include non-verbal sounds like (laughs) or (coughs).")
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text Input",
                placeholder="[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices.",
                lines=5
            )
            audio_prompt = gr.Audio(
                label="Audio Prompt (Optional)",
                type="filepath"
            )
            
            with gr.Row():
                with gr.Column():
                    temperature = gr.Slider(
                        minimum=0.1, 
                        maximum=2.0, 
                        value=1.0, 
                        step=0.1, 
                        label="Temperature"
                    )
                    top_p = gr.Slider(
                        minimum=0.1, 
                        maximum=1.0, 
                        value=0.9, 
                        step=0.05, 
                        label="Top P"
                    )
                with gr.Column():
                    cfg_scale = gr.Slider(
                        minimum=1.0, 
                        maximum=7.0, 
                        value=3.0, 
                        step=0.5, 
                        label="CFG Scale"
                    )
                    max_tokens = gr.Number(
                        label="Max Tokens (leave empty for default)",
                        value=None
                    )
            
            generate_btn = gr.Button("Generate Audio", variant="primary")
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(label="Generated Audio")
            
    generate_btn.click(
        fn=run_inference,
        inputs=[
            text_input,
            audio_prompt,
            temperature,
            top_p,
            cfg_scale,
            max_tokens
        ],
        outputs=audio_output
    )
    
    # Add examples
    gr.Examples(
        examples=[
            ["[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on GitHub or Hugging Face."],
            ["[S1] The quick brown fox jumps over the lazy dog. [S2] But why would a fox jump over a dog? [S1] It's just a pangram used to test typefaces. [S2] Oh, I see. That makes sense."],
            ["[S1] Did you hear about the new restaurant on the moon? (pauses) Great food, but no atmosphere! (laughs)"]
        ],
        inputs=text_input
    )

    gr.Markdown("## Tips for Best Results")
    gr.Markdown("""
    - Use [S1], [S2], etc. to indicate different speakers
    - Include non-verbal cues like (laughs), (coughs), (pauses) for more natural speech
    - When using an audio prompt, place the transcript of the prompt audio before your main script
    - For voice consistency, either use an audio prompt or keep the same temperature and seed
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
