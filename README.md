# Semantic-Video-Synthesis-with-BERT
Semantic Video Synthesis with BERT uses BERT's contextual understanding to generate short, meaningful videos from text prompts. It extracts key semantic elements from input text and translates them into coherent visual sequences, enabling automated, intelligent video creation.


run_whisper.py
# Step 1: Load Whisper model
# Step 2: Transcribe audio with .transcribe 
# Step 3: Load full audio for slicing
# Step 4: Save each segment as WAV and print MFCC shape
- # Save segment to WAV file for next processing
- # Save metadata inside the same directory
- # Extract MFCC features (optional print)


trainmodel
# mt_win (mid-term window size):
This is the length of the longer window (in seconds) used to extract stable features over chunks of audio.
For example, 1.0 means features are computed over 1-second chunks.

# mt_step (mid-term step size):
How much the mid-term window slides forward each time (in seconds).
For example, 0.5 means the window moves forward by 0.5 seconds between feature calculations, allowing some overlap.

# st_win (short-term window size):
Inside each mid-term window, features like MFCCs are extracted on smaller "short-term" windows.
0.05 means 50 milliseconds — a common value to capture the fine-grained spectral details.

# st_step (short-term step size):
How much the short-term window slides forward each time inside the mid-term window (in seconds).
0.025 means 25 milliseconds step, so short-term windows overlap by 50%.
