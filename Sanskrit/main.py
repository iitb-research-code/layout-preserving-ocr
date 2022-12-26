import layoutparser as lp
import pandas as pd
import numpy as np
import cv2
import os
try:
 from PIL import Image
except ImportError:
 import Image
import pytesseract
from pdf2image import convert_from_path
import sys
from pdfreader import SimplePDFViewer
import subprocess
import json
from pathlib import Path
from uuid import uuid4
from math import floor
import sys
import requests
import tarfile
from os import path
from PIL import Image
from PIL import ImageFont, ImageDraw
from glob import glob
from matplotlib import pyplot as plt

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import argparse
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

# Import some common libraries

import numpy as np
import os, json, cv2, random

# Import some common detectron2 utilities

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

from detectron2.structures import BoxMode
import yaml
from detectron2.data.datasets import register_coco_instances


tessdata_dir_config = r'--tessdata-dir "indic-parser/configs/tessdata"'
os.environ["TESSDATA_PREFIX"] = 'indic-parser/configs/tessdata'
languages=pytesseract.get_languages(config=tessdata_dir_config)

input_lang='san_iitb'

ocr_agent = lp.TesseractAgent(languages=input_lang)

LEVELS = {
    'page_num': 1,
    'block_num': 2,
    'par_num': 3,
    'line_num': 4,
    'word_num': 5
}

def infer_layout(input_image_path):
  custom_config = "custom_labels_weights.yml"
  with open(custom_config, 'r') as stream:
      custom_yml_loaded = yaml.safe_load(stream)

  config_list = list(custom_yml_loaded['WEIGHT_CATALOG'].keys()) + list(custom_yml_loaded['MODEL_CATALOG'].keys())
  # print("config_list is ",config_list)

  config_filePath = "configs/layout_parser_configs"
  
  config_files = []
  for cfile in config_list:
      config_files.append(cfile)
  config_name = 'Sanskrit_PubLayNet_faster_rcnn'

  # Capture model weights

  if config_name.split('_')[0] == 'Sanskrit':
      core_config = config_name.replace('Sanskrit_', '')
      config_file = config_filePath + '/' + custom_yml_loaded['MODEL_CATALOG'][core_config]
      model_weights = custom_yml_loaded['WEIGHT_CATALOG'][config_name]
      label_mapping = custom_yml_loaded['LABEL_CATALOG']["Sanskrit_Finetuned"]

  else:
      config_file = config_filePath + '/' + custom_yml_loaded['MODEL_CATALOG'][config_name]
      yaml_file = open(config_file)
      parsed_yaml_file = yaml.load(yaml_file, Loader = yaml.FullLoader)
      model_weights = parsed_yaml_file['MODEL']['WEIGHTS']
      dataset = config_name.split('_')[0]
      label_mapping = custom_yml_loaded['LABEL_CATALOG'][dataset]

  label_list = list(label_mapping.values())
  confidence_threshold = 0.7
  # print(" ")

  # Set custom configurations

  cfg = get_cfg()
  cfg.merge_from_file(config_file)
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold # set threshold for this model
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(list(label_mapping.keys()))

  # print("ROI Heads is taken as",cfg.MODEL.ROI_HEADS.NUM_CLASSES)

  cfg.MODEL.WEIGHTS =  model_weights
  cfg.MODEL.DEVICE='cpu'
  # Get predictions

  predictor = DefaultPredictor(cfg)
  im = cv2.imread(input_image_path)
  im_name = input_image_path.split("/")[-1]
  im_shape = im.shape[:2]
  # cv2.imwrite(f"{output_dir}/{im_name}", im)
  outputs = predictor(im)

  # Save predictions

  dataset_name = 'data'
  DatasetCatalog.clear()
  MetadataCatalog.get(f"{dataset_name}_infer").set(thing_classes=label_list)
  layout_metadata = MetadataCatalog.get(f"{dataset_name}_infer")
  # print("Metadata is ",layout_metadata)

  v = Visualizer(im[:, :, ::-1],
                      metadata=layout_metadata, 
                      scale=0.5
        )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  ans = out.get_image()[:, :, ::-1]
  im = Image.fromarray(ans)
  # img_name = 'image_with_predictions.jpg'
  # im.save(f"{output_dir}/{img_name}")

  # extracting, bboxes, scores and labels

  img = Image.open(input_image_path)
  instances = outputs["instances"].to("cpu")
  boxes = instances.pred_boxes.tensor.tolist()
  scores = instances.scores.tolist()
  labels = instances.pred_classes.tolist()
  layout_info = {}

  count = {}
  for i in range(len(label_list)):
    count[label_list[i]] = 0

  for score, box, label in zip(scores, boxes, labels):
    x_1, y_1, x_2, y_2 = box
    label_name = label_mapping[label]
    count[label_name] += 1
    l_new = label_name+str(count[label_name])
    info_data = {"box":box, "confidence": score}
    layout_info[l_new] = info_data
    #print(str(l_new) + ":",box)

  # storing the labels and corresponding bbox coordinates in a json
  layout_info_sort = {k: v for k, v in sorted(layout_info.items(), key=lambda item: item[1]["box"][1], reverse=True)}
  
  return img, layout_info, im_name, im_shape

def create_image_url(filepath):
  """
  Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
  if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
  Otherwise you can build links like /data/upload/filename.png to refer to the files
  """
  filename = os.path.basename(filepath)
  return f'http://localhost:8081/{filename}'

def convert_to_ls(image, tesseract_output, per_level='block_num'):
  """
  :param image: PIL image object
  :param tesseract_output: the output from tesseract
  :param per_level: control the granularity of bboxes from tesseract
  :return: tasks.json ready to be imported into Label Studio with "Optical Character Recognition" template
  """
  image_width, image_height = image.size
  per_level_idx = LEVELS[per_level]
  results = []
  all_scores = []
  for i, level_idx in enumerate(tesseract_output['level']):
    if level_idx == per_level_idx:
      bbox = {
        'x': 100 * tesseract_output['left'][i] / image_width,
        'y': 100 * tesseract_output['top'][i] / image_height,
        'width': 100 * tesseract_output['width'][i] / image_width,
        'height': 100 * tesseract_output['height'][i] / image_height,
        'rotation': 0
      }

      words, confidences = [], []
      for j, curr_id in enumerate(tesseract_output[per_level]):
        if curr_id != tesseract_output[per_level][i]:
          continue
        word = tesseract_output['text'][j]
        confidence = tesseract_output['conf'][j]
        words.append(word)
        if confidence != '-1':
          confidences.append(float(confidence / 100.))

      text = ' '.join((str(v) for v in words)).strip()
      if not text:
        continue
      region_id = str(uuid4())[:10]
      score = sum(confidences) / len(confidences) if confidences else 0
      bbox_result = {
        'id': region_id, 'from_name': 'bbox', 'to_name': 'image', 'type': 'rectangle',
        'value': bbox}
      transcription_result = {
        'id': region_id, 'from_name': 'transcription', 'to_name': 'image', 'type': 'textarea',
        'value': dict(text=[text], **bbox), 'score': score}
      results.extend([bbox_result, transcription_result])
      all_scores.append(score)

  return {
    'data': {
      'ocr': create_image_url(image.filename)
    },
    'predictions': [{
      'result': results,
      'score': sum(all_scores) / len(all_scores) if all_scores else 0
    }]
  }

def create_hocr(image_path, languages, linput, output_path):
  pytesseract.pytesseract.run_tesseract(image_path, output_path, extension="jpg", lang=languages[linput], config="--psm 4 -c tessedit_create_hocr=1")
 
def get_layout_data(input_image_path):
  img, layout_info, im_name, im_shape = infer_layout(input_image_path)
  #sorting layout_info by y_1 coordinate
  hocr_data = {}
  layout_info_sort = {k: v for k, v in sorted(layout_info.items(), key=lambda item: item[1]["box"][1], reverse=True)}
  # with open(f'{output_dir}/output-ocr.txt', 'w') as f:
  for label, info_dict in layout_info_sort.items():
    img_cropped = img.crop(info_dict["box"])
    res = ocr_agent.detect(img_cropped)
    # f.write(res)
    q = layout_info_sort[label]
    q["text"] = res
    hocr_data[label] = q

  # print(hocr_data)
  hocr_sorted_data = {k: v for k, v in sorted(hocr_data.items(), key=lambda item: item[1]["box"][1])}
  header = f'''
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
  "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html lang="en" xml:lang="en" xmlns="http://www.w3.org/1999/xhtml">
 <head>
  <title>
  </title>
  <meta content="text/html;charset=utf-8" http-equiv="Content-Type"/>
  <meta content="tesseract 5.0.0-alpha-20201231-256-g73a32" name="ocr-system"/>
  <meta content="ocr_page ocr_carea ocr_par ocr_line ocrx_word ocrp_wconf" name="ocr-capabilities"/>
 </head>
 <body>
  <div class="ocr_page" id="page_1" title='image "{im_name}"; bbox 0 0 {im_shape[0]} {im_shape[1]}; ppageno 0'>\n
  '''
  # print(list(hocr_sorted_data.items()))

  for i, item in enumerate(list(hocr_sorted_data.items())):
    label=item[0]
    k=item[1]['text']
    bbox = " ".join([str(floor(value)) for value in hocr_sorted_data[label]["box"]])
    x1=floor(hocr_sorted_data[label]["box"][0])
    y1=floor(hocr_sorted_data[label]["box"][1])
    x2=floor(hocr_sorted_data[label]["box"][2])
    y2=floor(hocr_sorted_data[label]["box"][3])
    k = k[:-2]
    if label.find('Text') != -1:
      c = 'sent'
      sent = f'   <span class="ocr_sent" title="bbox {bbox};">{k}</span>\n'
    elif label.find('Image') != -1:
      c = 'image'
      img_block = img.crop([x1,y1,x2,y2])
      img_file_name = f"{label}.jpg"
      img_block = img_block.save(img_file_name)
      # cv2.imwrite(img_file_name, img_block)
      sent = f'   <img class="ocr_image" src={img_file_name} title="bbox {bbox};">\n'
    elif label.find('Table') != -1:
      c = 'table'
      tab_block = img.crop([x1,y1,x2,y2])
      tab_file_name = f"{label}.jpg"
      tab_block = img_block.save(tab_file_name)
      # cv2.imwrite(tab_file_name, tab_block)
      sent = f'   <img class="ocr_tab" src={tab_file_name} title="bbox {bbox};">\n'
    else:
      sent = f'   <span class="ocr_sent" title="bbox {bbox};">{k}</span>\n'
    header += sent
  footer = '''
    </div>
 </body>
</html>
'''
  header += footer

  return header

if __name__ == "__main__":
  img_path = "/content/indic-parser/test_img/7.jpeg"
  output = get_layout_data(img_path)
  print(output)
