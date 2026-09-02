# Ear: On-Device Mandarin Pronunciation Tutor via CTC

Local setup and implementation based on Simon Edwardsson's blog post:
**[A 9M-parameter Mandarin pronunciation tutor](https://simedw.com/2026/01/31/ear-pronunication-via-ctc/)**

---

## 🌟 Features

- **Tiny 9M Conformer Model**: ~11 MB INT8 quantized ONNX model (`tiny_int8.onnx`).
- **Pinyin + Tone as First-Class Tokens**: 1,254 token vocabulary mapping every Pinyin syllable + tone (1–5).
- **Viterbi CTC Forced Alignment**: Pinpoints syllable timings and evaluates confidence per character without auto-correcting errors.
- **Silence Filtering**: Ignores blank leading/trailing frames during scoring so natural pauses don't degrade confidence scores.
- **Tone Sandhi Rules**: Built-in support for standard Mandarin sandhi (e.g. 3-3 tone sandhi, 不 `bu4` before 4th tone, 一 `yi1` before 4th or 1/2/3 tones).
- **Web & CLI Interfaces**:
  1. **Interactive Web App**: In-browser client-side evaluation with microphone input and audio playback.
  2. **Python CLI & API**: Evaluate audio files (WAV, MP3, FLAC, AIFF) directly from terminal.

---

## 🚀 Quick Start

### 1. Python Virtual Environment

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

### 2. Run the Local Web App (Interactive Browser UI)

Start the local server:
```bash
./server.py
# or: python3 server.py --port 8000
```

Open your browser at:
👉 **[http://localhost:8000](http://localhost:8000)**

You can:
- Enter any Chinese sentence or click the example phrases.
- Tap the microphone button or press `Spacebar` to record.
- Listen to your recording playback.
- View real-time character-by-character pronunciation and tone feedback.

---

### 3. Run via Python Command-Line Interface (CLI)

Evaluate any audio file against a target Chinese sentence:

```bash
# Terminal table format
./cli.py eval samples/test_friend.wav --sentence "他是我的朋友"

# JSON format
./cli.py eval samples/test_friend.wav --sentence "他是我的朋友" --json
```

Convert Chinese sentences to expected CTC tokens:
```bash
./cli.py pinyin "他是我的朋友"
```

---

## 📁 Project Structure

```
.
├── tiny_int8.onnx         # 9M parameter INT8 quantized Conformer CTC model (~14 MB)
├── vocab.json             # 1,254 Pinyin+Tone vocabulary tokens
├── index.html             # Web application interface
├── audio-processor.js     # Web Audio API 80-mel filterbank & STFT extraction
├── pinyin.js              # Web Pinyin conversion utilities
├── ctc.js                 # Log-softmax & Viterbi forced alignment in JS
├── scoring.js             # Syllable scoring & Sandhi tolerance logic in JS
├── ear_model.py           # Complete Python model, alignment & scoring module
├── cli.py                 # Python command-line evaluation tool
├── server.py              # Local HTTP server with CORS/WASM headers
├── requirements.txt       # Python dependencies
└── samples/               # Sample audio files for testing
```

---

## 🧠 Architecture Overview

```
Audio (16kHz PCM)
       │
       ▼
STFT Power Spectrum (n_fft=400, hop=160, Hann window)
       │
       ▼
80-dim Log Mel Filterbank + Per-utterance CMVN
       │
       ▼
Conformer CTC Encoder (9M parameters, INT8)
       │
       ▼
Log-Softmax Frame Probabilities (T x 1256)
       │
       ▼
Viterbi CTC Forced Alignment (Target Tokens)
       │
       ▼
Non-Blank Frame Filtering & Conditioned Probabilities
       │
       ▼
Tone Sandhi Post-Processing & Per-Syllable Grading
```
