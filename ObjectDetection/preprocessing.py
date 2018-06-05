import os
import cv2
import numpy as np
from keras.utils import Sequence


class BatchGenerator(Sequence):
    def __init__(self, x_set, y_set, config):
        self.x, self.y = x_set, y_set
        self.config = config

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.config["batch_size"])))

    def __getitem__(self, item):
        # bounds
        l_bound = item * self.config["batch_size"]
        r_bound = item * self.config["batch_size"]

        if r_bound > len(self.x):
            r_bound = len(self.x)
            l_bound = r_bound - self.config["batch_size"]

        istance_count = 0

        # Prepare the placeholders for the desired input/output tensors.
        # Inputs have two dimension, image height and width.
        # Output have the grid dimensions (height and width), the maximum number of box to search and the number of
        # parameter of each label, that is 4 coordinates, the object score and 'c' class scores, where 'c' is the number
        # of classes in the dataset.

        x_batch = np.zeros((r_bound - l_bound, self.config["image_h"], self.config["image_w"], 1))
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config["true_box_buffer"]))
        y_batch = np.zeros((r_bound - l_bound, self.config["grid_h"], self.config["grid_w"], self.config["nb_boxes"],
                            4 + 1 + len(self.config["labels"])))

        for train_istance in self.x:
            img = cv2.imread(train_istance)


    def load_image(train_istance):
            return