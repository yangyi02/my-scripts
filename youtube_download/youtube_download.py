# Lint as: python3
"""Download a group of Youtube play lists."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pytube
import time

from absl import app
from absl import flags
from absl import logging

flags.DEFINE_string('input_file', './youtube_links.txt', 'Youtue links file')
flags.DEFINE_string('output_path', './videos', 'Output video path')
flags.DEFINE_string('tmp_path', '/tmp/youtube_download', 'Intermediate path')
flags.DEFINE_integer('resolution', 1080, 'video resolution')

FLAGS = flags.FLAGS


def parse_lines(lines):
  youtube_links = []
  for line in lines:
    items = line.strip().split(' ')
    link = {}
    link['name'] = items[0]
    link['playlist_url'] = items[1]
    youtube_links.append(link)
  return youtube_links


def parse_youtube_id(url):
  youtube_id = url.split('v=')[1]
  return youtube_id


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  youtube_links = parse_lines(open(FLAGS.input_file).readlines())
  logging.info('%s', repr(youtube_links))
  for link in youtube_links:
    playlist = pytube.Playlist(link['playlist_url'])
    playlist.populate_video_urls()
    logging.info('%s', repr(playlist.video_urls))
    for video_url in playlist.video_urls:
      youtube_id = parse_youtube_id(video_url)
      output_path = os.path.join(FLAGS.output_path, link['name'])
      success_file_name = os.path.join(output_path, youtube_id + '.txt')
      if os.path.exists(success_file_name):
        final_file_name = os.path.join(output_path, youtube_id + '.mp4')
        logging.info('already download %s to %s', video_url, final_file_name)
        continue

      streams = pytube.YouTube(video_url).streams
      import pdb; pdb.set_trace()
      video = streams.filter(type='video', subtype='mp4', res='%dp' % FLAGS.resolution).first()
      logging.info('%s', repr(video))
      audio = streams.filter(type='audio', subtype='mp4').first()
      logging.info('%s', repr(audio))

      tmp_path = os.path.join(FLAGS.tmp_path, link['name'])
      if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
      import pdb; pdb.set_trace()
      video_file_name = youtube_id +'_video'
      logging.info('download video to %s/%s.mp4', tmp_path, video_file_name)
      video.download(output_path=tmp_path, filename=video_file_name)
      logging.info('video download done')
      audio_file_name = youtube_id +'_audio'
      logging.info('download audio to %s/%s.mp4', tmp_path, audio_file_name)
      audio.download(output_path=tmp_path, filename=audio_file_name)
      logging.info('audio download done')

      logging.info('merge video audio')
      video_file_name = os.path.join(tmp_path, video_file_name + '.mp4')
      audio_file_name = os.path.join(tmp_path, audio_file_name + '.mp4')
      output_path = os.path.join(FLAGS.output_path, link['name'])
      if not os.path.exists(output_path):
        os.makedirs(output_path)
      final_file_name = os.path.join(output_path, youtube_id + '.mp4')
      cmd = ('ffmpeg -y -i %s -i %s -c:a copy %s' %
             (video_file_name, audio_file_name, final_file_name))
      logging.info('%s', cmd)
      os.system(cmd)

      with open(success_file_name, 'w') as handle:
        handle.write('success')
      logging.info('success download %s to %s', video_url, final_file_name)

      logging.info('wait 10 mintues')
      time.sleep(600)


if __name__ == '__main__':
  app.run(main)
