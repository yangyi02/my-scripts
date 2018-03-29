#!/usr/bin/python

import base64
import json
import random
import requests
import sys
import time

url = "https://api.dxdy.co/en/SpeechFrontEnd/Transcribe"
packet_size_ms = 200

if len(sys.argv) < 3:
    print "Usage: {} filename.pcm sample_rate [verbose=true/false]".format(sys.argv[0])
    sys.exit(1)

filename = sys.argv[1]
sample_rate = int(sys.argv[2])
verbose = False

if len(sys.argv) == 4:
    verbose = sys.argv[3] == "true"

session_id = str(random.randrange(0, 2**64))  # Choose a random session number
with open(filename) as f:
    audio = f.read()

# Chunk into 100ms samples.
chunk_len = 2 * sample_rate * packet_size_ms / 1000 # 100ms at 16000hz, 16bit, mono

last_ind = xrange(0, len(audio), chunk_len)[-1]

# Create session request session.
with requests.Session() as sfess:
    t_start_global = time.time()
    for sequence_num, i in enumerate(xrange(0, len(audio), chunk_len)):
        chunk = audio[i:i+chunk_len]
        base64_chunk = base64.b64encode(chunk)
        is_last = i == last_ind

        packet = {
                "token": "harmancn",
                "session_id": session_id,
                "sequence_id": sequence_num,
                "format": "PCM",
                "language": "zh_CN",
                "sample_rate": sample_rate,
                "audio": base64_chunk,
                "is_last": is_last,
        }

        json_msg = json.dumps(packet)

        if verbose:
            print "request:", json_msg
        else:
            print "sent packet {}".format(sequence_num)

        t_send = time.time()
        r = sfess.post(url, data=json_msg)
        t_tot = time.time() - t_send

        response = r.text.encode('utf-8')
        if verbose:
            print "response:", response
            print "Delay = %dms" % int(t_tot * 1000)

        if not is_last:
            # sleep packet_size_ms to simulate real time speech (500us buffer for python time)
            time.sleep(max(0, 1. * packet_size_ms / 1000 - t_tot - 0.0005))

print "Transcription:", json.loads(response)["transcription"][0]
print "Latency: {}ms".format(int(t_tot * 1000))
print "Total time: {}".format(time.time() - t_start_global, 1. * len(audio) / 2 / sample_rate)

