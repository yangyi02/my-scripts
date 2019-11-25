#!/bin/bash

# Prerequisite
# 1. ffmpeg
# 2. pip install opencv-python
# 3. pip install pydub

ffmpeg -i 1.mp4 # This gives you frame rate

# Make sure you give the correct frame rate
python split_video.py --input_video_file=1.mp4 --output_segment_path=./video_segments --fps=23.98 --segment_length=50
