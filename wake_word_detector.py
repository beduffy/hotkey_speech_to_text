import sys
import time
import struct
import subprocess as sp
import shutil
import numpy as np
import threading
import torch
import torchaudio
import silero_vad
from pathlib import Path
import tempfile
import wave

# --- Wake Word Detection Dependencies ---
# This typically requires a dedicated library. Popular choices include:
# - pvporcupine: Commercial, but offers a free tier. Very accurate.
#                (https://picovoice.ai/platform/porcupine/)
#                Install: pip install pvporcupine
# - pocketsphinx: Open source, might require more tuning.
#                 (https://github.com/cmusphinx/pocketsphinx)
# - snowboy (archive): Was popular, now less maintained, might still work.
#                      (https://github.com/Kitt-AI/snowboy)
#
# You'll also likely need an audio input library like sounddevice or PyAudio
# to feed audio data to the wake word engine.
# Install: pip install sounddevice (or PyAudio)
# -----------------------------------------

# Example using placeholder for pvporcupine (adjust if using another engine)
try:
    import pvporcupine
    import sounddevice as sd
except ImportError as e:
    print(f"Error importing dependencies: {e}", file=sys.stderr, flush=True)
    print("  Please install required packages: pip install pvporcupine sounddevice silero-vad torch torchaudio", file=sys.stderr, flush=True)
    # Optionally exit or disable wake word feature
    # sys.exit(1)

# Whisper import (similar to main.py)
try:
    import whisper
except ImportError:
    whisper = None # Handled later


class WakeWordDetector:
    # Define states
    STATE_WAITING = "WAITING_FOR_WAKE_WORD"
    STATE_RECORDING = "RECORDING_COMMAND"

    def __init__(self, access_key=None, library_path=None, model_path=None, keyword_paths=None, sensitivities=None):
        """
        Initializes the wake word engine.

        Args:
            access_key (str): Access key for Porcupine (if using). Get from Picovoice Console.
            library_path (str): Path to Porcupine's dynamic library file.
            model_path (str): Path to Porcupine's model file.
            keyword_paths (list[str]): List of paths to keyword model files (.ppn).
            sensitivities (list[float]): Sensitivities for each keyword (0.0 to 1.0).
        """
        # TODO: Replace with actual initialization for your chosen engine
        print("Initializing Wake Word Detector...", flush=True)
        if not all([access_key, keyword_paths, sensitivities]):
             print("Warning: Missing Porcupine configuration (access_key, keyword_paths, sensitivities)", file=sys.stderr, flush=True)
             print("         Wake word detection will likely fail.", file=sys.stderr, flush=True)

        self._access_key = access_key
        self._library_path = library_path
        self._model_path = model_path
        self._keyword_paths = keyword_paths
        self._sensitivities = sensitivities

        self._handle = None # Placeholder for engine instance
        self._audio_stream = None # For sounddevice stream

        # VAD related variables
        self._vad_model = None
        self._vad_utils = None # For get_speech_timestamps etc.
        self._current_state = self.STATE_WAITING
        self._command_audio_buffer = [] # Store chunks of command audio
        self._is_speaking = False

        # Whisper model
        self._whisper_model = None

        # Example parameters (adjust based on engine requirements)
        self.frame_length = None
        self.sample_rate = None

        try:
            self._setup_engine()
            self._setup_vad()
            self._setup_whisper()
            if self._handle:
                 self._setup_audio_stream()
            print("Wake Word Detector Initialized.", flush=True)
        except Exception as e:
             print(f"Error during Wake Word Detector initialization: {e}", file=sys.stderr, flush=True)
             self.stop() # Clean up any partial initialization


    def _setup_engine(self):
        """Sets up the specific wake word engine (e.g., Porcupine)."""
        if not pvporcupine:
            print("pvporcupine library not imported.", file=sys.stderr, flush=True)
            return

        try:
            self._handle = pvporcupine.create(
                access_key=self._access_key,
                library_path=self._library_path, # Often None, library finds itself
                model_path=self._model_path,     # Often None, library finds itself
                keyword_paths=self._keyword_paths,
                sensitivities=self._sensitivities)

            # Get required audio parameters from the initialized engine
            self.sample_rate = self._handle.sample_rate
            self.frame_length = self._handle.frame_length
            print(f"Porcupine initialized. Sample Rate: {self.sample_rate}, Frame Length: {self.frame_length}", flush=True)
            print(f"Watching for keywords: {self._keyword_paths}", flush=True)

        except pvporcupine.PorcupineError as e:
             print(f"Error initializing Porcupine: {e}", file=sys.stderr, flush=True)
             if "invalid AccessKey" in str(e):
                 print("Please ensure your PICOVOICE_ACCESS_KEY in the script is correct.", file=sys.stderr, flush=True)
             elif "Unable to find keyword file" in str(e):
                  print(f"Please ensure the keyword file path(s) are correct: {self._keyword_paths}", file=sys.stderr, flush=True)
             self._handle = None
             raise # Re-raise the exception to be caught in __init__
        except Exception as e:
            print(f"An unexpected error occurred during Porcupine initialization: {e}", file=sys.stderr, flush=True)
            self._handle = None
            raise # Re-raise the exception

    def _setup_audio_stream(self):
        """Sets up the audio input stream using sounddevice."""
        if not sd:
            print("Sounddevice library not imported.", file=sys.stderr, flush=True)
            return

        assert self.sample_rate is not None # This is Porcupine's required rate (16kHz)
        assert self.frame_length is not None

        # We MUST capture at a rate the device supports (e.g., 48kHz)
        # and then resample down to self.sample_rate (16kHz) for Porcupine.
        self.capture_sample_rate = 48000
        # Calculate the frame length needed at the capture rate to get
        # self.frame_length samples after downsampling by factor 3 (48k -> 16k)
        self.capture_frame_length = self.frame_length * (self.capture_sample_rate // self.sample_rate)

        print(f"Attempting to capture at {self.capture_sample_rate} Hz (frame length {self.capture_frame_length}) for resampling...", flush=True)

        try:
            self._audio_stream = sd.InputStream(
                 samplerate=self.capture_sample_rate,
                 blocksize=self.capture_frame_length, # Read blocks suitable for resampling
                 device=1,                    # Use explicit device index
                 channels=1,
                 dtype='int16',
                 )
            self._audio_stream.start()
            print("Audio stream opened successfully at capture rate.", flush=True)
        except Exception as e:
            print(f"Error opening audio stream at {self.capture_sample_rate} Hz: {e}", file=sys.stderr, flush=True)
            self._audio_stream = None
            raise # Re-raise the exception

    def run(self):
        """
        Runs the main loop, listening for wake words and commands.
        """
        if not self._handle or not self._audio_stream:
            # Initialization should have handled this, but check just in case.
            print("Error: Wake word engine or audio stream not ready.", file=sys.stderr, flush=True)
            return None

        # --- VAD Configuration --- 
        # Silero VAD works best on 16kHz chunks, but can handle others.
        # We need to feed it audio compatible with its requirements.
        # Since we capture at 48kHz, we'll resample for VAD too.
        vad_sample_rate = 16000 # Silero VAD preferred rate
        vad_resample_factor = self.capture_sample_rate // vad_sample_rate
        # VAD works on chunks; window size examples: 256, 512, 768, 1024, 1536 samples (at 16kHz)
        # Let's use a similar frame size as Porcupine for consistency
        vad_window_size = 512 # samples at vad_sample_rate
        vad_window_size_capture = vad_window_size * vad_resample_factor
        # Threshold for speech probability (adjust as needed)
        vad_threshold = 0.5
        # --- End VAD Config --- 

        print(f"Starting main loop... State: {self._current_state}", flush=True)
        try:
            # Buffer to hold samples between reads, ensuring we have enough to process
            buffer = np.array([], dtype=np.int16)
            resample_factor = self.capture_sample_rate // self.sample_rate # Should be 3 (48k / 16k)

            while True:
                # Read audio frames captured at the higher sample rate
                # Read a larger chunk to better accommodate VAD processing
                read_length = self.capture_frame_length * 4 # Read a few porcupine frames worth
                pcm_capture, overflowed = self._audio_stream.read(read_length)

                if overflowed:
                    print("Warning: Audio buffer overflowed during capture.", file=sys.stderr, flush=True)

                # Append new samples to the buffer
                new_samples = pcm_capture.flatten()
                buffer = np.concatenate((buffer, new_samples))

                # --- State-Based Processing --- 

                if self._current_state == self.STATE_WAITING:
                     # Process frames as long as we have enough samples in the buffer for Porcupine
                     process_frame_len_capture = self.frame_length * resample_factor
                     while len(buffer) >= process_frame_len_capture:
                         frame_to_process_capture = buffer[:process_frame_len_capture]
                         buffer = buffer[process_frame_len_capture:] # Consume from buffer
                         pcm_resampled = frame_to_process_capture[::resample_factor]
                         if len(pcm_resampled) != self.frame_length:
                             print(f"Warning: Porcupine frame length mismatch. {len(pcm_resampled)} != {self.frame_length}", file=sys.stderr, flush=True)
                             continue
                         result = self._handle.process(pcm_resampled)
                         if result >= 0:
                             print(f"\nWake word detected! ({self._keyword_paths[result]})", flush=True)
                             self._current_state = self.STATE_RECORDING
                             self._command_audio_buffer = [] # Clear buffer for command
                             self._is_speaking = False # Reset speaking flag
                             # Optional: Play start sound
                             print("Listening for command...", flush=True)
                             break # Exit inner loop to handle VAD in outer loop
                     # If buffer has less than a frame left, keep it for next read

                elif self._current_state == self.STATE_RECORDING:
                     # Need enough *new* samples in the buffer for VAD check
                     # Let's process based on the new samples read in this iteration
                     if len(new_samples) > 0 and self._vad_model:
                         # Resample the *new* audio chunk for VAD
                         vad_samples_resampled = new_samples[::vad_resample_factor]
                         # Convert to torch tensor
                         audio_tensor = torch.from_numpy(vad_samples_resampled).float()
                         # Get speech probability
                         speech_prob = self._vad_model(audio_tensor, vad_sample_rate).item()

                         # --- Simple VAD Logic --- 
                         if speech_prob > vad_threshold:
                             print(".", end='' , flush=True) # Indicate speech detected
                             self._is_speaking = True
                             # Append the ORIGINAL 48kHz audio to the command buffer
                             self._command_audio_buffer.append(new_samples)
                         elif self._is_speaking:
                             # Speech was happening, but now it stopped (silence)
                             print("\nSilence detected, processing command.", flush=True)
                             # Combine buffered command audio
                             command_audio = np.concatenate(self._command_audio_buffer)
                             self._command_audio_buffer = [] # Clear buffer
                             self._is_speaking = False

                             # --- Process the Command Audio --- 
                             if command_audio.size > 0:
                                 # TODO: Integrate whisper transcription here
                                 print(f"  Command audio recorded ({command_audio.size / self.capture_sample_rate:.2f}s). Transcribing...", flush=True)
                                 # Example: save and transcribe
                                 temp_wav_path = Path(tempfile.gettempdir()) / "temp_command.wav"
                                 self.save_wav(temp_wav_path, command_audio, self.capture_sample_rate)
                                 # Add transcription logic here (needs whisper model access)
                                 if self._whisper_model:
                                     try:
                                         # Transcribe the 48kHz audio directly
                                         result = self._whisper_model.transcribe(str(temp_wav_path))
                                         text = result["text"].strip()
                                         print(f"  Transcription: {text}", flush=True)
                                         # TODO: Act on transcription (e.g., pass to GPT, run command)
                                     except Exception as e:
                                         print(f"  Transcription failed: {e}", file=sys.stderr, flush=True)
                                 else:
                                     print("  Whisper model not loaded, cannot transcribe.", file=sys.stderr, flush=True)
                             else:
                                 print("  No command audio captured after wake word.", flush=True)

                             # --- Switch back to waiting state --- 
                             self._current_state = self.STATE_WAITING
                             print(f"\nSwitching back to {self._current_state}...", flush=True)
                             # Ensure buffer doesn't grow indefinitely if VAD misses silence
                             buffer = np.array([], dtype=np.int16)
                     # Ensure buffer doesn't grow indefinitely if only silence is detected
                     if len(buffer) > self.capture_sample_rate * 10: # Keep max 10 secs in buffer
                          print("Warning: Clearing excessive buffer during silence.", file=sys.stderr, flush=True)
                          buffer = buffer[-self.capture_sample_rate * 5:] # Keep last 5 secs

        except KeyboardInterrupt:
            print("\nStopping wake word listener (KeyboardInterrupt).", flush=True)
            return None
        except Exception as e:
            print(f"Unexpected error during wake word detection: {e}", file=sys.stderr, flush=True)
            return None

    def stop(self):
        """Releases resources used by the wake word detector."""
        print("Stopping Wake Word Detector...", flush=True)

        if self._audio_stream is not None:
            try:
                if not self._audio_stream.closed:
                     self._audio_stream.stop()
                     self._audio_stream.close()
            except Exception as e:
                 print(f"Error closing audio stream: {e}", file=sys.stderr, flush=True)
            finally:
                 self._audio_stream = None

        if self._handle is not None:
            try:
                self._handle.delete()
            except Exception as e:
                 print(f"Error deleting Porcupine handle: {e}", file=sys.stderr, flush=True)
            finally:
                 self._handle = None

        print("Wake Word Detector stopped.")

    def _setup_vad(self):
        """Loads the Silero VAD model."""
        try:
             # Load the VAD model
             # Note: This downloads the model on first run
             self._vad_model, self._vad_utils = silero_vad.load_silero_vad(force_reload=False)
             print("Silero VAD model loaded.", flush=True)
        except Exception as e:
             print(f"Error loading Silero VAD model: {e}", file=sys.stderr, flush=True)
             print("  Ensure torch and torchaudio are installed correctly.", file=sys.stderr, flush=True)
             self._vad_model = None
             # Decide if we should raise or just disable VAD
             # raise

    def _setup_whisper(self):
        """Loads the Whisper model."""
        if not whisper:
            print("Error: openai-whisper library not imported. Cannot transcribe.", file=sys.stderr, flush=True)
            print("  Install it with: pip install openai-whisper", file=sys.stderr, flush=True)
            return

        # Reuse model loading logic (consider making this configurable)
        try:
            model_version = "base.en" # Or small.en, medium.en, large
            print(f"Loading local Whisper model ({model_version})...", file=sys.stderr, flush=True)
            self._whisper_model = whisper.load_model(model_version)
            print("Whisper model loaded.", flush=True)
        except Exception as exc:
            print(f"Failed to load Whisper model: {exc}", file=sys.stderr, flush=True)
            self._whisper_model = None

    # Helper to save WAV (similar to main.py, but takes path object)
    def save_wav(self, path: Path, signal: np.ndarray, samplerate: int):
        """Write *signal* to a WAV file at *path*."""
        # Normalize to int16 range if it's float
        if signal.dtype == np.float32 or signal.dtype == np.float64:
            signal = np.clip(signal, -1.0, 1.0)
            signal = (signal * 32767).astype(np.int16)
        elif signal.dtype != np.int16:
            raise ValueError("Signal must be int16 or float for WAV saving")

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(samplerate)
            wf.writeframes(signal.tobytes())
        print(f"Saved WAV to {path}", flush=True)


# Example usage (for testing this file directly)
if __name__ == '__main__':
    # --- Configuration (replace with your actual details) ---
    # Obtain AccessKey from Picovoice Console (https://console.picovoice.ai/)
    # PICOVOICE_ACCESS_KEY = None # "YOUR_ACCESS_KEY_HERE"
    PICOVOICE_ACCESS_KEY = "hIqd9l1cMSx4eed3Yy8aaKUnk9aHYneJGYAldp9OsPeOGflgONKVnA==" # "YOUR_ACCESS_KEY_HERE"

    # Assuming you have downloaded keyword files (e.g., 'Hey-Computer_en_linux_v2_2_0.ppn')
    # and placed them in the same directory or specified the correct path.
    KEYWORD_FILE_PATHS = ['ppns/Hi-Johnny_en_linux_v3_0_0.ppn'] # ["path/to/your/keyword.ppn"]
    SENSITIVITIES = [0.5] # [0.5] # Adjust sensitivity (0.0 to 1.0)

    # Optional: Specify library/model paths if not in default location
    LIBRARY_PATH = None
    # MODEL_PATH = 'ppns/Hi-Johnny_en_linux_v3_0_0.ppn'
    MODEL_PATH = None
    # --- End Configuration ---

    if not PICOVOICE_ACCESS_KEY or not KEYWORD_FILE_PATHS or not SENSITIVITIES:
         print("********************************************************************************")
         print("Warning: PICOVOICE_ACCESS_KEY, KEYWORD_FILE_PATHS, or SENSITIVITIES not set.", flush=True)
         print("         Please configure them in wake_word_detector.py for actual use.", flush=True)
         print("         Running with placeholder logic.", flush=True)
         print("********************************************************************************")
         # Use placeholder values for basic structure testing
         PICOVOICE_ACCESS_KEY = "dummy"
         KEYWORD_FILE_PATHS = ["dummy.ppn"]
         SENSITIVITIES = [0.5]


    detector = WakeWordDetector(
        access_key=PICOVOICE_ACCESS_KEY,
        library_path=LIBRARY_PATH,
        model_path=MODEL_PATH,
        keyword_paths=KEYWORD_FILE_PATHS,
        sensitivities=SENSITIVITIES
    )

    try:
        # In a real application, you might call this in a loop or thread
        # The run method now handles the continuous loop
        detector.run()

    finally:
        detector.stop() 