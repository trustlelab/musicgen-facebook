from transformers import pipeline
import scipy
from tqdm import tqdm

# Initialize the text-to-audio pipeline
synthesiser = pipeline("text-to-audio", "facebook/musicgen-large", device="cuda")

# Define the length of the sequence to be generated
max_length = 32000  # You might need to adjust this value for a 2-minute song

# Display progress in the terminal
with tqdm(total=max_length, desc="Generating music", unit="tokens") as pbar:
    def progress_callback(current_length):
        pbar.update(current_length - pbar.n)

    # Generate the music with progress tracking
    music = synthesiser(
        "lo-fi music with a soothing melody", 
        forward_params={"do_sample": True, "max_length": max_length},
        progress_bar=progress_callback  # Call the progress callback during generation
    )

# Save the generated music to a WAV file
scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
