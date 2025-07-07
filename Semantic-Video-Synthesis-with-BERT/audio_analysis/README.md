# Semantic-Video-Synthesis-with-BERT
Semantic Video Synthesis with BERT uses BERT's contextual understanding to generate short, meaningful videos from text prompts. It extracts key semantic elements from input text and translates them into coherent visual sequences, enabling automated, intelligent video creation.


General 
run_whisper.py
here firts we upload an audio of few minutes and then the audio id broken down into many segments with its timestamps and MFcc coefficinet and stored in teh folder segments 
like this 
Segment 1:
  Filename: C:/Users/Asus/Desktop/BERT/Semantic-Video-Synthesis-with-BERT\segments\segment_1.wav
  Text:  What you should look out on buying a dog, the legs and gait.
  Start: 3.0 sec, End: 8.0 sec
  MFCC mean coefficients: [-499.417      119.90342    -15.318568    33.135014     6.981997
   15.962022    -6.0865345    9.906875    -9.253828   -13.007545
   -3.3960207    4.2992954  -16.02198  ]

Segment 2:
  Filename: C:/Users/Asus/Desktop/BERT/Semantic-Video-Synthesis-with-BERT\segments\segment_2.wav
  Text:  Notice how the dog walks.
  Start: 8.0 sec, End: 10.0 sec
  MFCC mean coefficients: [-493.38602    118.28104    -13.712041    38.035603   -12.724239
   12.846589   -10.387521    12.468615    -8.923319    -5.598548
   -0.7728255    5.172528   -15.544069 ]
 and the last one segment conatins the metadata



















































































For neha
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
