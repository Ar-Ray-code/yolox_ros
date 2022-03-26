#!/bin/bash

# if $1 is empty
if [ -z "$1" ]; then
    echo "Usage: $0 <target-model>"
    echo "Target-Models :"
    echo "yolox_tiny, yolox_nano, yolox_s, yolox_m, yolox_l, all"
    exit 1
fi
MODEL=$1
SCRIPT_DIR=$(cd $(dirname $0); pwd)

echo $MODEL
if [ "$MODEL" = "all" ]; then
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_tiny.pth -P $SCRIPT_DIR
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_nano.pth -P $SCRIPT_DIR
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_s.pth -P $SCRIPT_DIR
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_m.pth -P $SCRIPT_DIR
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/yolox_l.pth -P $SCRIPT_DIR
else
    wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.0/$MODEL.pth -P $SCRIPT_DIR
fi
