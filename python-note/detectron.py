import random
import cv2
import json
import os
import detectron2
import torch
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from google.colab.patches import cv2_imshow
import numpy as np
from detectron2.utils.logger import setup_logger
setup_logger()

!wget http: // images.cocodataset.org/val2017/000000439715.jpg - q - O input.jpg
im = cv2.imread("./input.jpg")
cv2_imshow(im)
