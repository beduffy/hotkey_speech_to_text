import sys
import time
import struct
import subprocess as sp
import shutil
import numpy as np

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
    print("  Please install required packages: pip install pvporcupine sounddevice", file=sys.stderr, flush=True)
    # Optionally exit or disable wake word feature
    # sys.exit(1)


class WakeWordDetector:
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

        # Example parameters (adjust based on engine requirements)
        self.frame_length = None
        self.sample_rate = None

        try:
            self._setup_engine()
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

    def wait_for_wake_word(self):
        """
        Listens continuously and returns the index of the detected keyword
        once a wake word is detected. Returns None if stopped or error occurs.
        """
        if not self._handle or not self._audio_stream:
            # Initialization should have handled this, but check just in case.
            print("Error: Wake word engine or audio stream not ready.", file=sys.stderr, flush=True)
            return None

        print(f"Listening for wake word(s)...", flush=True)
        try:
            # Buffer to hold samples between reads, ensuring we have enough to process
            buffer = np.array([], dtype=np.int16)
            resample_factor = self.capture_sample_rate // self.sample_rate # Should be 3 (48k / 16k)

            while True:
                # Read audio frames captured at the higher sample rate
                pcm_capture, overflowed = self._audio_stream.read(self.capture_frame_length)

                if overflowed:
                    print("Warning: Audio buffer overflowed during capture.", file=sys.stderr, flush=True)

                # Append new samples to the buffer
                buffer = np.concatenate((buffer, pcm_capture.flatten()))

                # Process frames as long as we have enough samples in the buffer
                # We need enough samples at the capture rate to produce one frame at the target rate
                process_frame_len_capture = self.frame_length * resample_factor

                while len(buffer) >= process_frame_len_capture:
                    # Extract samples needed for one Porcupine frame from the buffer
                    frame_to_process_capture = buffer[:process_frame_len_capture]
                    # Remove processed samples from the buffer
                    buffer = buffer[process_frame_len_capture:]

                    # Resample by taking every Nth sample
                    pcm_resampled = frame_to_process_capture[::resample_factor]

                    # Ensure the resampled frame has the correct length expected by Porcupine
                    if len(pcm_resampled) != self.frame_length:
                        print(f"Warning: Resampled frame length mismatch. Expected {self.frame_length}, got {len(pcm_resampled)}", file=sys.stderr, flush=True)
                        continue # Skip this frame

                    # Pass the resampled frame to Porcupine
                    result = self._handle.process(pcm_resampled)

                    # Check if a keyword was detected
                    if result >= 0:
                        print(f"Detected keyword index: {result} ({self._keyword_paths[result]})", flush=True)
                        return result # Return the index of the detected keyword

            # If loop exits, stream became inactive without detection (shouldn't happen in while True)
            print("Audio stream loop exited unexpectedly.", file=sys.stderr, flush=True)
            return None

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
        detected_keyword_index = detector.wait_for_wake_word()

        if detected_keyword_index is not None:
             print(f"Wake word detected (index {detected_keyword_index})! Triggering action...", flush=True)
             # TODO: Add logic here to start recording in main.py or trigger other actions
        else:
            print("Wake word listener finished without detection (or was stopped).", flush=True)

    finally:
        detector.stop() 