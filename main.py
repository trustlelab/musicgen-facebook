from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-large", device="cuda")

# music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True})

# Generate the music
music = synthesiser(
    "lo-fi music with a soothing melody", 
    forward_params={"do_sample": True, "max_length": 32000}  # Adjust max_length for 2 minutes
)
scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])
