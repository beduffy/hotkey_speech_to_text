import whisper
from openai import OpenAI
import sounddevice as sd
import soundfile as sf
import numpy as np
import subprocess
import os


# TODO Add a wake word (like "Hey Computer") using tools like Porcupine
# TODO Add command confirmation before execution
# TODO Expand the system prompt with more command examples
# TODO Add command history
# TODO Add a stop command ("exit" or "stop listening"
# TODO better continuous listening loop
# TODO to get to bash action of wave, from my laptop i probably do need ros action hmm
# TODO initial easier version would be to ssh -c "ssh -c 'docker run action'

# Run ls command over SSH and get output
# output = subprocess.check_output(['ssh', 'pi@192.168.178.49', 'ls'], text=True)
# print(f"Remote directory contents:\n{output}")





class VoiceCommandExecutor:
    def __init__(self):
        # Initialize Whisper (using the smallest model)
        print("Loading Whisper model...")
        # self.whisper_model = whisper.load_model("tiny")
        self.whisper_model = whisper.load_model("base.en")
        # self.whisper_model = whisper.load_model("medium")  # slow to load the data hmm
        print("Whisper model loaded successfully!")
        
        # Recording settings - try common sample rates
        self.sample_rates = [44100, 48000, 16000, 22050]  # Common rates to try
        self.duration = 5  # seconds

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # self.model = "gpt-4o"
        # self.model = "gpt-4o-mini"
        self.model = "gpt-3.5-turbo"
        # TODO how to do system prompt?
        
        # Find working sample rate
        # self.sample_rate = self._get_working_sample_rate()
        # if not self.sample_rate:
        #     raise RuntimeError("Could not find a working sample rate")
        self.sample_rate = 48000
        print(f"Using sample rate: {self.sample_rate}")
    

    def _get_working_sample_rate(self):
        """Test different sample rates and return the first working one."""
        for rate in self.sample_rates:
            try:
                # Try to record a tiny bit of audio
                sd.rec(int(0.1 * rate), samplerate=rate, channels=1)
                sd.wait()
                print(f"Rate {rate} worked!")
                return rate
            except Exception as e:
                print(f"Rate {rate} failed: {e}")
        return None
    

    def record_audio(self):
        """Record audio with error handling."""
        print("Recording... Speak now!")
        try:
            recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                blocking=True  # Simpler than using wait()
            )
            print('finished recording')
            return recording
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
    

    def transcribe_audio(self, audio):
        print('transcribing audio')
        # Save temporary audio file
        sf.write("temp_recording.wav", audio, self.sample_rate)
        
        # Transcribe
        result = self.whisper_model.transcribe("temp_recording.wav")
        os.remove("temp_recording.wav")
        print('finished transcribing audio')
        
        return result["text"]
    
    
    def get_shell_command(self, text):
        # TODO how to not run system prompt every time? keep memory?
        system_prompt = """You are a command line interface translator. Your only job is to convert natural language to shell commands.
        ONLY respond with the exact shell command, nothing else. No explanations, no quotes.
        
        Examples:
        Input: "list directory"
        Output: ls
        
        Input: "change directory"
        Output: cd
        
        Input: "show current path"
        Output: pwd
        
        Input: "go to downloads folder"
        Output: cd Downloads
        
        Input: "make a new directory called test"
        Output: mkdir test
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Convert this to a shell command: {text}"}
                ],
                temperature=0,  # Make responses more deterministic
                max_tokens=10   # We only need a short response
            )
            
            # Extract just the command
            command = response.choices[0].message.content.strip()
            
            # Remove any quotes or explanation text
            command = command.replace('"', '').replace("'", "")
            
            # Basic validation - if command contains multiple lines or explanatory text, take first line
            command = command.split('\n')[0]
            
            return command
            
        except Exception as e:
            print(f"Error generating command: {e}")
            return "echo 'Error generating command'"


    def execute_command(self, command):
        # List of allowed commands for safety
        ALLOWED_COMMANDS = ['ls', 'pwd', 'cd', 'echo', 'mkdir', 'cat', 'touch']
        
        # Basic safety check
        command_base = command.split()[0]
        if command_base not in ALLOWED_COMMANDS:
            print(f"Command '{command}' not in allowed list")
            return
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                capture_output=True
            )
            print(f"Output: \n{result.stdout}")
            if result.stderr:
                print(f"Error: {result.stderr}")
        except Exception as e:
            print(f"Error executing command: {e}")
    

    def run(self):
        while True:
            # Record audio
            audio = self.record_audio()
            
            # Convert speech to text
            text = self.transcribe_audio(audio)
            print(f"Whisper heard: {text}")
            
            # Get shell command
            command = self.get_shell_command(text)
            print(f"Executing: {command}")
            
            # Execute command
            self.execute_command(command)


if __name__ == "__main__":
    executor = VoiceCommandExecutor()
    executor.run()
