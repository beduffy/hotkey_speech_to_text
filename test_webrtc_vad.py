import sounddevice as sd
import numpy as np
import webrtcvad
import time
import sys
import collections
import time # Add time for monotonic clock

# --- Configuration ---
CAPTURE_SAMPLE_RATE = 48000 # Rate your microphone supports
VAD_SAMPLE_RATE = 16000     # Rate WebRTC VAD works with (8k, 16k, 32k)
DEVICE_INDEX = 1            # Your microphone index
VAD_AGGRESSIVENESS = 3      # 0 (least aggressive) to 3 (most aggressive)
VAD_FRAME_MS = 30           # Frame duration for VAD (10, 20, or 30 ms)
VAD_FRAME_SAMPLES = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)
VAD_BYTES_PER_FRAME = VAD_FRAME_SAMPLES * 2 # 16-bit PCM = 2 bytes/sample
CAPTURE_RESAMPLE_FACTOR = CAPTURE_SAMPLE_RATE // VAD_SAMPLE_RATE
# --- End Configuration ---

print("Initializing WebRTC VAD...", flush=True)
try:
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    print(f"WebRTC VAD Initialized (Rate: {VAD_SAMPLE_RATE}Hz, Frame: {VAD_FRAME_MS}ms, Aggressiveness: {VAD_AGGRESSIVENESS}).", flush=True)
except Exception as e:
    print(f"Error initializing WebRTC VAD: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    sys.exit(1)

# Ring buffer to hold resampled audio for VAD processing
ring_buffer = collections.deque(maxlen=VAD_FRAME_SAMPLES * 2) # Hold a bit extra

# State variables
currently_speaking = False
silence_start_time = None
last_status_printed = ""

# --- Audio Callback ---
def audio_callback(indata, frames, time_info, status):
    global ring_buffer, currently_speaking, silence_start_time, last_status_printed

    if status:
        print(f"Audio stream status: {status}", file=sys.stderr, flush=True)
        return

    try:
        # Use the captured audio directly (already int16)
        pcm_capture = indata[:, 0]

        # Resample for VAD - simple method for now
        pcm_vad_int16 = pcm_capture[::CAPTURE_RESAMPLE_FACTOR]

        # Add resampled audio to the ring buffer
        ring_buffer.extend(pcm_vad_int16)

        # Determine if any frame in this chunk contains speech
        chunk_contains_speech = False
        frames_processed_in_chunk = 0
        while len(ring_buffer) >= VAD_FRAME_SAMPLES:
            frames_processed_in_chunk += 1
            frame_samples = np.array([ring_buffer.popleft() for _ in range(VAD_FRAME_SAMPLES)])
            frame_bytes = frame_samples.tobytes()

            try:
                if vad.is_speech(frame_bytes, VAD_SAMPLE_RATE):
                    chunk_contains_speech = True
                    # Optional: If we find speech, no need to check rest of frames in this chunk?
                    # break # Could uncomment this for slight efficiency gain
            except Exception as e:
                print(f"\nError processing VAD frame: {e}", file=sys.stderr, flush=True)

        # Update state based on whether speech was detected in the *entire chunk*
        current_time = time.monotonic() # Use monotonic clock for duration
        status_message = ""

        if chunk_contains_speech:
            if not currently_speaking:
                status_message = "\rSpeech started...                                        "
                currently_speaking = True
                silence_start_time = None # Reset silence timer
            else:
                 # Still speaking, maybe just print listening indicator
                 status_message = "\rListening... (Speech ongoing)                        "
        else: # No speech detected in this chunk
            if currently_speaking:
                status_message = "\rSpeech stopped. Silence starting...                     "
                currently_speaking = False
                silence_start_time = current_time
            elif silence_start_time is not None:
                silence_duration = current_time - silence_start_time
                status_message = f"\rSilence detected for {silence_duration:.2f} seconds            "
            else:
                 # Initial state or just started, remain silent
                 status_message = "\rWaiting for speech...                              "
                 if silence_start_time is None: # Initialize timer on first silent chunk
                     silence_start_time = current_time

        # Print status only if it changed to avoid excessive printing
        if status_message and status_message != last_status_printed:
            print(status_message, end='', flush=True)
            last_status_printed = status_message

    except Exception as e:
        print(f"\nError in audio callback: {type(e).__name__}: {e}", file=sys.stderr, flush=True)

# --- Main Execution ---
print(f"Attempting to open audio stream (Device: {DEVICE_INDEX}, Rate: {CAPTURE_SAMPLE_RATE}Hz)...", flush=True)
try:
    # Capture at a rate supported by the device
    with sd.InputStream(samplerate=CAPTURE_SAMPLE_RATE,
                         blocksize=0, # Use default blocksize
                         device=DEVICE_INDEX,
                         channels=1,
                         dtype='int16',
                         callback=audio_callback):
        print("Audio stream opened. Listening for speech (Press Ctrl+C to stop)...", flush=True)
        # Initialize status print
        print("\rWaiting for speech...                              ", end='', flush=True)
        last_status_printed = "\rWaiting for speech...                              "

        while True:
            time.sleep(0.1) # Keep main thread alive

except KeyboardInterrupt:
    print("\nStopping test.")
except Exception as e:
    print(f"\nError setting up audio stream: {type(e).__name__}: {e}", file=sys.stderr, flush=True) 