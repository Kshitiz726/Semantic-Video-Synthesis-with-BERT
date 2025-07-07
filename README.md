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


# The Above Prototype was refined with transnet scene analysis

[2025-July- 07] : Transnet Scene Analysis 

-> Same directory as above
-> The video segment where there is scene change, threshold = 0.6, has been clipped out
-> Scene Analysis requires manually setting up TransnetV2, (you cant directly install it via pip, if you try) you can try out cloning

```bash
git clone https://github.com/soCzech/TransNetV2
```
```bash
cd TransNetV2
```

Make sure tensorflow, matplotlib and numpy are preinstalled via pip 


"While running it on your computer, you might have to see the functions and set it up according to the directory


After installing transnetv2, you'll have to place the transnet_scene_extractor.py program inside the directory "transnetv2"

than run this 

```bash
python transnet_scene_extractor.py ../input_video/input.mp4
```

The audio analysis modulde built by Neha will be Integrated to calculate the MFCC score of  respective scene changed audio under extracted/videoname/Audio , and time stamp , scores, transcribed text and their respective MFCC scores,  will be recorded in scene_summary.csv & scene_summary.json for analysis.......