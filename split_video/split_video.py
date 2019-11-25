# Lint as: python3
"""Split a video into a list of equal segments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

import cv2
import numpy as np
import os
import sys
from pydub import AudioSegment

flags.DEFINE_string('input_video_file', './1.mp4', 'input video file')
flags.DEFINE_string('output_segment_path', './video_segments', 'output segment path')
flags.DEFINE_float('fps', 23.98, 'video fps')
flags.DEFINE_integer('segment_length', 50, 'number of frames per segment')
flags.DEFINE_string('tmp_dir', './tmp', 'temporary path')

FLAGS = flags.FLAGS


def main(argv):
  del argv

  if not os.path.exists(FLAGS.tmp_dir):
    os.makedirs(FLAGS.tmp_dir)

  audio_file = os.path.join(FLAGS.tmp_dir, 'audio.wav')
  cmd_str = 'ffmpeg -i %s %s' % (FLAGS.input_video_file, audio_file)
  os.system(cmd_str)

  # Check the total number of frames
  cap = cv2.VideoCapture(FLAGS.input_video_file)
  num_frames = 0
  while True:
    ret, frame = cap.read()
    if not ret: break
    height, width, layers = frame.shape
    size = (width, height)
    num_frames += 1
  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()
  print ('total number of frames: ', num_frames)
  print ('frame size: (%d, %d)' % (height, width))

  # Generate frame index for each video segment
  indices = []
  for i in range(num_frames):
    if i % FLAGS.segment_length == 0:
      indices.append([i])
    else:
      indices[-1].append(i)
  print ('segment indices: ', indices)

  # Prepare to write video and audio segments
  cap = cv2.VideoCapture(FLAGS.input_video_file)
  fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
  audio = AudioSegment.from_wav(audio_file)

  for i in range(len(indices)):
    # Write video segments (no audio)
    video_name = os.path.join(FLAGS.tmp_dir, '%.3d.mp4' % i)
    print ('Writing video segment to: ', video_name)
    out = cv2.VideoWriter(video_name, fourcc, FLAGS.fps, size)
    for j in indices[i]:
      # Capture frame-by-frame
      ret, frame = cap.read()
      out.write(frame)
    out.release()

    # Write audio segments
    audio_name = os.path.join(FLAGS.tmp_dir, '%.3d.wav' % i)
    t1 = indices[i][0]
    t2 = indices[i][-1] + 1
    t1 = float(t1) / FLAGS.fps * 1000.0  # Works in milliseconds
    t2 = float(t2) / FLAGS.fps * 1000.0
    t1 = int(np.round(t1))
    t2 = int(np.round(t2))
    audio_segment = audio[t1:t2]
    print ('Writing audio segment to: ', audio_name)
    audio_segment.export(audio_name, format='wav')

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()

  # Prepare to generate the final video segments with audio
  if not os.path.exists(FLAGS.output_segment_path):
    os.makedirs(FLAGS.output_segment_path)

  for i in range(len(indices)):
    # Merge video and audio segments
    video_name = os.path.join(FLAGS.tmp_dir, '%.3d.mp4' % i)
    audio_name = os.path.join(FLAGS.tmp_dir, '%.3d.wav' % i)
    segment_name = os.path.join(FLAGS.output_segment_path, '%.3d.mp4' % i)
    cmd_str = 'ffmpeg -i %s -i %s -c:v copy -c:a aac -strict experimental %s'
    cmd_str = cmd_str % (video_name, audio_name, segment_name)
    print (cmd_str)
    os.system(cmd_str)

  print ('Done')

if __name__ == '__main__':
  app.run(main)
