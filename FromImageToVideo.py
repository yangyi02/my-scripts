import numpy as np
from PIL import Image
import cv2

image_list = open('imagelist.txt').readlines()

# Load one image to get image size
im = Image.open('part_seg/' + image_list[0].strip())
height, width, layers = np.array(im).shape
print height, width, layers

# Create video from a list of images
video = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (width, height))
for file_name in image_list:
    file_name = file_name.strip()
    im = Image.open('part_seg/' + file_name)
    video.write(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))

video.release()
