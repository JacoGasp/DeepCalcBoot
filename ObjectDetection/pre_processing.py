import os
import cv2
from utils import *
import numpy as np
from keras.utils import Sequence
import xml.etree.ElementTree as ET


def parse_annotation(ann_dir, img_dir, labels=None):
    if labels is None:
        labels = []
    all_imgs = []
    seen_labels = {}

    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}

        tree = ET.parse(ann_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1

                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]

    return all_imgs, seen_labels


class BatchGenerator(Sequence):
    def __init__(self, images, config, norm=None, grayscale=False):
        self.generator = None
        self.images = images
        self.config = config
        self.norm = norm
        self.grayscale = grayscale
        self.anchors = [BoundBox(0, 0, config["anchors"][2 * i], config["anchors"][2 * i + 1]) for i in
                        range(len(config["anchors"]) // 2)]

    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.config["batch_size"])))

    def num_classes(self):
        """
        Return the number of classes.
        """
        return len(self.config["labels"])

    def size(self):
        """
        Return the number of images.
        """
        return len(self.images)

    def load_image(self, train_instance):
        if self.grayscale:
            img = cv2.imread(train_instance["filename"], cv2.IMREAD_GRAYSCALE)
            # Transform the image shape from (H_PIXELS, W_PIXELS) to (H_PIXELS, W_PIXELS, 1)
            img = np.expand_dims(img, axis=-1)
        else:
            img = cv2.imread(train_instance["filename"])

        return img

    def __getitem__(self, item):
        # bounds
        l_bound = item * self.config["batch_size"]
        r_bound = item * self.config["batch_size"]

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config["batch_size"]

        instance_count = 0

        # Prepare the placeholders for the desired input/output tensors.
        # Inputs have two dimension, image height and width.
        # Output have the grid dimensions (height and width), the maximum number of box to search and the number of
        # parameter of each label, that is 4 coordinates, the object score and 'c' class scores, where 'c' is the number
        # of classes in the dataset.

        x_batch = np.zeros((r_bound - l_bound, self.config["image_h"], self.config["image_w"], 1))
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config["true_box_buffer"]))
        y_batch = np.zeros((r_bound - l_bound, self.config["grid_h"], self.config["grid_w"], self.config["nb_max_box"],
                            4 + 1 + len(self.config["labels"])))

        for train_instance in self.images[l_bound:r_bound]:
            img = self.load_image(train_instance)
            h, w, c = img.shape
            all_objs = train_instance["object"]

            # construct output from object's x, y, w, h
            true_box_index = 0

            # Fix object's position and size
            for obj in all_objs:
                for attr in ["xmin", "xmax"]:
                    obj[attr] = int(obj[attr] * self.config["image_w"] / w)
                    obj[attr] = max(min(obj[attr], self.config["image_w"]), 0)
                for attr in ["ymin", "ymax"]:
                    obj[attr] = int(obj[attr] * self.config["image_h"] / h)
                    obj[attr] = max(min(obj[attr], self.config["image_h"]), 0)

                if obj["xmax"] > obj["xmin"] and obj["ymax"] > obj["ymin"] and obj["name"] in self.config["labels"]:
                    center_x = 0.5 * (obj["xmin"] + obj["xmax"])
                    center_x = center_x / (self.config["image_w"] / self.config["grid_w"])
                    center_y = 0.5 * (obj["ymin"] + obj["ymax"])
                    center_y = center_y / (self.config["image_h"] / self.config["grid_h"])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config["grid_w"] and grid_y < self.config["grid_h"]:
                        obj_idx = self.config["labels"].index(obj["name"])

                        center_w = (obj["xmax"] - obj["xmin"]) / self.config["image_w"] / self.config["grid_w"]
                        center_h = (obj["ymax"] - obj["ymin"]) / self.config["image_h"] / self.config["grid_h"]

                        box = [center_x, center_y, center_w, center_h]

                        # Find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou = -1

                        shifted_box = BoundBox(0, 0, center_w, center_h)

                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou = bbox_iou(shifted_box, anchor)
                            if max_iou < iou:
                                best_anchor = i
                                max_iou = iou

                        # Assign ground truth x, y, w, h, confidence and class probabilities to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4] = 1.0
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5 + obj_idx] = 1

                        # Assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        true_box_index += 1
                        true_box_index = true_box_index % self.config["true_box_buffer"]

            # Assign input image to x_batch
            if self.norm:
                x_batch[instance_count] = self.norm(img)
            else:
                x_batch[instance_count] = img
        instance_count += 1
        print(len(b_batch))

        return [x_batch, b_batch], y_batch
