import os
from os.path import join as ojoin

import pandas as pd
from PIL import Image, ImageDraw
from tqdm import *

ROOT_DIR = os.environ['ROOT_DIR']
DATA_FOLDER = ojoin(ROOT_DIR, 'darknet', 'data', 'tw')
classes = ["manhole", "other"]


class YOLODataset(object):

    def __init__(self, data_folder=DATA_FOLDER):
        self.data_folder = data_folder

    def convert(self, size, box):
        '''
        Convert from x0,y0,x1,y1 to YOLO format
        '''
        dw = 1. / size[0]
        dh = 1. / size[1]
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return [x, y, w, h]

    def arr2nlist(self, data):
        return [list(f) for f in list(data)]

    def read_csv(self, filename):
        data = pd.read_csv(filename, index_col=0).values
        data = self.arr2nlist(data)
        return data

    def convert_bboxes(self, p, img_id):
        im = Image.open(ojoin(p, '{}.jpg'.format(img_id)))
        bboxes = self.read_csv(ojoin(p, '{}.csv'.format(img_id)))
        new_bbox = []
        for bbox in bboxes:
            bbox = self.convert(im.size, bbox)
            # Add a zero for the class. Only 1 here
            new_bbox.append([0] + bbox)
        new_bbox = pd.DataFrame(new_bbox)
        # Dump the file to a csv file with a space separator
        fname = ojoin(p, 'labels', '{}.txt'.format(img_id))
        new_bbox.to_csv(fname, index=None, header=None, sep=" ")

    def create_annotations(self):
        '''
        Create the annotation for yolo
        '''
        for split in ['train', 'validation', 'test']:
            print(split)
            p = ojoin(self.data_folder, split)
            if not os.path.isdir(ojoin(p, 'labels')):
                os.mkdir(ojoin(p, 'labels'))

            img_ids = [os.path.splitext(f)[0]
                       for f in os.listdir(p) if f.endswith('.jpg')]
            path_imgs = []
            for img_id in tqdm(img_ids):
                path_imgs.append(ojoin(p, '{}.jpg'.format(img_id)))
                self.convert_bboxes(p, img_id)

            data = pd.DataFrame(path_imgs)
            data.to_csv(ojoin(self.data_folder, '{}.txt'.format(
                split)), index=None)

if __name__ == '__main__':

    data = YOLODataset()
    data.create_annotations()
