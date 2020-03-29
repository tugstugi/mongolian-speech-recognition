#!/usr/bin/env sh

BASEDIR=$(dirname $0)
cd $BASEDIR/..

docker run -it --rm \
  --gpus all \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v `pwd`:/workspace/mongolian-speech-recognition \
  mongolian-speech-recognition:latest bash
