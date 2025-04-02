import warnings
import torch
import onnx
import subprocess
import tensorflow as tf
import os
from ultralytics import YOLO

from ultralytics import YOLO
model = YOLO("/content/yolov10n.pt")
model.export(format="tflite")