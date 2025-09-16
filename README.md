---
title: Speech Bot
emoji: 🏃
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.42.0
python_version: "3.13"
app_file: app.py
pinned: false
license: apache-2.0
---

# Speech Coach

Record a short clip in the browser → **transcribe** with Faster-Whisper → get **LLM feedback** via OpenRouter → hear a **Piper TTS** reply.  
No local LLM needed for users.

## Features
- Browser mic input (Gradio `Audio`)
- STT via `faster-whisper` (CPU-friendly)
- Coaching feedback via **OpenRouter** (free model defaults)
- TTS reply via **Piper** (ONNX voice)
- One-click pipeline: **Analyze (STT → Coach → TTS)**
- Manual debug buttons (Play, Transcribe, Coach, Speak)

## Project Structure
├─ app.py # Gradio app (this repo’s main file)  
├─ requirements.txt  
├─ packages.txt # system deps for audio (Spaces)  
└─ models/ # Piper voice files (.onnx + .json)  

## How to Run

1. Open the app: [Speech Coach on Hugging Face](https://huggingface.co/spaces/Mashroor14/Speech-Coach)
2. Click the **microphone (🎙️)** icon.
3. Allow your browser to access the microphone.
4. Press **Record**, then speak in English (longer clips take longer to process).
5. When finished, click **Analyze**.
6. Wait for the analysis to complete (processing time depends on audio length).
7. Click **Play** on the reply audio to hear the TTS feedback.

## Notes

* For faster processing, close or minimize other background apps while using the UI.
* This is an **early** prototype; speed, UI, voice quality, and overall user experience will be improved in future updates.
