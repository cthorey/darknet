import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

import pandas as pd
import sys
from tqdm import *

ROOT_DIR = os.environ['ROOT_DIR']
sys.path.append(ROOT_DIR)

DATA = os.path.join(ROOT_DIR, 'darknet', 'data')

sets = [('2012', 'train'), ('2012', 'val'),
        ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(year, image_id):

    in_file = open(
        join(DATA, 'VOCdevkit/VOC%s/Annotations/%s.xml' % (year, image_id)))
    out_file = open(join(DATA, 'VOCdevkit/VOC%s/labels/%s.txt' %
                         (year, image_id)), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')


for year, image_set in sets:
    print(year, image_set)
    p = join(DATA, 'VOCdevkit', 'VOC{}'.format(year), 'labels')
    if not os.path.exists(p):
        os.makedirs(p)
    image_path = join(DATA, 'VOCdevkit', 'VOC{}'.format(
        year), 'ImageSets', 'Main', '{}.txt'.format(image_set))
    image_ids = open(image_path).read().strip().split()
    imgstack = []
    for image_id in tqdm(image_ids):
        name = join(DATA, 'VOCdevkit', 'VOC{}'.format(
            year), 'JPEGImages', '{}.jpg'.format(image_id))
        imgstack.append(name)
        convert_annotation(year, image_id)

    data = pd.DataFrame(imgstack)
    data.to_csv(join(DATA, 'VOCdevkit', '{}_{}.txt'.format(
        year, image_set)), index=None)
