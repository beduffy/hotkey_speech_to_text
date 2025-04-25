import sounddevice as sd
import numpy as np
import torch
import silero_vad
import resampy
import time
import sys

# --- Configuration ---
CAPTURE_SAMPLE_RATE = 48000 # Rate your microphone supports
VAD_SAMPLE_RATE = 16000     # Rate Silero VAD expects
DEVICE_INDEX = 1            # Your microphone index
VAD_THRESHOLD = 0.5         # Speech probability threshold
# --- End Configuration ---

print("Loading Silero VAD model...", flush=True)
try:
    # Try forcing a redownload of the model
    vad_model, utils = silero_vad.load_silero_vad(force_reload=True)
    (get_speech_timestamps, _, read_audio, _, _) = utils
    print("Silero VAD model loaded.", flush=True)
except Exception as e:
    print(f"Error loading Silero VAD model: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    print("Ensure torch, torchaudio, and silero-vad are installed correctly.", file=sys.stderr, flush=True)
    sys.exit(1)

# --- Audio Callback ---
def audio_callback(indata, frames, time, status):
    if status:
        print(f"Audio stream status: {status}", file=sys.stderr, flush=True)
        return

    try:
        # Use the captured audio directly (already int16)
        pcm_capture = indata[:, 0]

        # Resample for VAD
        pcm_capture_float = pcm_capture.astype(np.float32)
        pcm_vad_float = resampy.resample(pcm_capture_float, CAPTURE_SAMPLE_RATE, VAD_SAMPLE_RATE, filter='kaiser_fast')

        # Convert to torch tensor for VAD model
        audio_tensor = torch.from_numpy(pcm_vad_float).float()

        # Get speech probability for the whole chunk
        # Note: Silero VAD works best on chunks > 30ms (at 16kHz, thats > 480 samples)
        # The blocksize from sounddevice callback should be sufficient.
        if len(audio_tensor) > 0:
            speech_prob = vad_model(audio_tensor, VAD_SAMPLE_RATE).item()
            is_speech = speech_prob >= VAD_THRESHOLD
            # Print status continuously
            print(f"\rVAD Prob: {speech_prob:.3f} | Speech: {is_speech} | Frames: {frames}   ", end='', flush=True)
        else:
            print("\rWarning: Received empty audio chunk?", end='', flush=True)

    except Exception as e:
        print(f"\nError in audio callback: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

# --- Main Execution ---
print(f"Attempting to open audio stream (Device: {DEVICE_INDEX}, Rate: {CAPTURE_SAMPLE_RATE}Hz)...", flush=True)
try:
    with sd.InputStream(samplerate=CAPTURE_SAMPLE_RATE,
                         blocksize=0, # Use default blocksize
                         device=DEVICE_INDEX,
                         channels=1,
                         dtype='int16',
                         callback=audio_callback):
        print("Audio stream opened. Listening for speech (Press Ctrl+C to stop)...", flush=True)
        while True:
            time.sleep(0.1) # Keep main thread alive

except KeyboardInterrupt:
    print("\nStopping test.")
except Exception as e:
    print(f"\nError setting up audio stream: {type(e).__name__}: {e}", file=sys.stderr, flush=True) 