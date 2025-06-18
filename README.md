# Semantic-Video-Synthesis-with-BERT
Dear friends, 
Please add your log here with the explanation and integration guidelines [No chatgpt]: 

[2025-june- 18] : Completed The preprocessing step of the video files
Worked on "Preprocessing"

->Tried out with a sample input video, (place it inside the input_video directory)
-> You can try out by placing any video here (in mp4 right now)
-> To run the preprocessing (right now) you need to be in the respective directory first
-> I've built the functions to extract the Frames, Audio, And metadata respectively, which you can use in your continuation
-> For now, working on isolated manner , you need to upload the input video into input_video directory, and than run the program under "preprocessing"  directory


```bash
python ffmpeg_extractor.py input_video/input.mp4 extracted/audio extracted/Frames extracted/metadata
```

Upon running, we can continue, with the audio analysis part as the workflow, using the audio under extracted/audio

You can use the functions built to do the audio analysis....