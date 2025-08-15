import gradio as gr
import numpy as np
import tempfile
import soundfile as sf
from faster_whisper import WhisperModel
import os, tempfile, shutil, subprocess, sys, time

# loads on first use to keep startup fast
_model = None
def get_model():
    global _model
    if _model is None:
        size = os.getenv("WHISPER_SIZE", "small")
        _model = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=2, num_workers=1)
    return _model

# Input Audio From the Mic
def passthrough(audio):
    if audio is None:
        return None
    sr, data = audio  # sr: int, data: float32 ndarray (mono)
    # ensure mono float32
    if data.ndim == 2:
        data = data[:, 0]
    data = data.astype(np.float32, copy=False)
    return (sr, data)

# STT (Speech-to-Text)
def transcribe(audio):
    
    if audio is None:
        return "No audio provided.", "", 0.0
    sr, data = audio
    if data.ndim == 2:
        data = data[:, 0]
    duration = float(len(data) / max(1, sr))

    # write temp wav for faster-whisper
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, data, sr, subtype="PCM_16")
        model = get_model()
        segments, info = model.transcribe(f.name, beam_size=5)

    text = "".join(s.text for s in segments).strip()
    if not text:
        text = "(no speech detected)"
    lang = f"{info.language} ({info.language_probability:.2f})" if info else "unknown"
    display = f"[lang: {lang}]  [dur: {duration:.2f}s]\n{text}"
    return display, text, duration

# LLM coach via OpenRouter
def coach_openrouter(transcript, duration):
    # Uses OpenRouter's OpenAI compatible API.
    # Requires env var: OPENROUTER_API_KEY
    # Free model Used: deepseek/deepseek-r1:free

    t = (transcript or "").strip()
    if not t or t == "(no speech detected)":
        return "I didn’t catch any speech. Try recording again."

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return "OPENROUTER_API_KEY is not set."

    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = (
        "You are a concise, supportive speech coach. "
        "Give feedback on the speaker's performance. Be sure to sound human. Do not include headers or classifiers such as 'Feedback:' or 'Summary'."
        "Keep under 200 words.\n\n"
        f"(Duration ~{duration:.1f}s)\n"
        f"Transcript:\n{t}"
    )

    try:
        chat = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=500,
        )

        msg_content = chat.choices[0].message.content
        if msg_content is None:
            return "(no content returned)"
        return msg_content.strip()

    except Exception as e:
        return f"OpenRouter error: {e}"

# Piper TTS (Text to Speech)
VOICE_MODEL  = os.getenv("PIPER_VOICE_MODEL", "models/en_US-lessac-medium.onnx")
VOICE_CONFIG = os.getenv("PIPER_VOICE_CONFIG", "models/en_US-lessac-medium.onnx.json")

# Function that determines how to invoke Piper TTS:
# If the 'piper' CLI is installed, return its path
# otherwise run Piper via the current Python interpreter as a module

def _piper_cmd():
    exe = shutil.which("piper")
    return [exe] if exe else [sys.executable, "-m", "piper"]

def _ensure_voice():
    for p in (VOICE_MODEL, VOICE_CONFIG):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing Piper voice file: {p}\n"
                "Place the .onnx + .json in models/"
            )

def speak(text):
    
    t = (text or "").strip()
    if not t:
        return None
    _ensure_voice()
    with tempfile.TemporaryDirectory() as td:
        out_wav = os.path.join(td, "tts.wav")
        cmd = _piper_cmd() + ["-m", VOICE_MODEL, "-c", VOICE_CONFIG, "-f", out_wav]
        subprocess.run(cmd, input=t, text=True, check=True)
        audio, sr = sf.read(out_wav, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return (sr, audio.astype(np.float32, copy=False))
    
def analyze(audio):

    t0 = time.time()
    disp, transcript, duration = transcribe(audio)
    t1 = time.time()

    feedback = coach_openrouter(transcript, duration)

    t2 = time.time()
    tts_audio = speak(feedback)
    t3 = time.time()

    timing = f"[timing] STT: {(t1-t0)*1000:.0f} ms • Coach: {(t2-t1)*1000:.0f} ms • TTS: {(t3-t2)*1000:.0f} ms"
    disp_time = f"{disp}\n\n{timing}"
    return disp_time, feedback, tts_audio

# UI
with gr.Blocks(title="Speech Coach") as demo:
    gr.Markdown("Speech Coach")

    # state (kept for saving transcript debug button)
    st_transcript = gr.State("")
    st_duration = gr.State(0.0)

    mic = gr.Audio(type="numpy", label="Record")

    analyze_btn = gr.Button("Analyze")

    out_text = gr.Textbox(label="Transcript", lines=6)
    out_feedback = gr.Textbox(label="Coach Feedback", lines=8)
    out_audio = gr.Audio(type="numpy", label="Reply Audio")

    gr.Markdown("---\n#### Debug / manual controls")

    with gr.Row():
        btn_play = gr.Button("Play Back")
        btn_stt = gr.Button("Transcribe")
        btn_coach_or = gr.Button("Coach (OpenRouter)")
        btn_tts = gr.Button("Speak (Piper)")

    # one-click pipeline
    analyze_btn.click(analyze, inputs=[mic], outputs=[out_text, out_feedback, out_audio])

    # debug/manual
    btn_play.click(passthrough, inputs=mic, outputs=out_audio)
    btn_stt.click(transcribe, inputs=mic, outputs=[out_text, st_transcript, st_duration])
    btn_coach_or.click(coach_openrouter, inputs=[st_transcript, st_duration], outputs=out_feedback)
    btn_tts.click(speak, inputs=out_feedback, outputs=out_audio)

if __name__ == "__main__":
    demo.launch()