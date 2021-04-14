import numpy as np

import tensorflow as tf
from tensorflow.python.keras.utils import data_utils
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

import random

from functools import partial

from SimCLR_Keras.preprocessing import preprocess_image

default_augmentation = partial(
    preprocess_image,
    operators=['crop', 'color_distort']
)

class DataGeneratorSimCLR(data_utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=16,
        subset="train",
        shuffle=True,
        info={},
        width=80,
        height=80,
        file_col='filename',
        augmentation_function=default_augmentation,
        preprocess_image=lambda x: x
    ):
        super().__init__()
        self.df = df
        self.indexes = np.asarray(self.df.index)
        self.batch_size = batch_size
        self.subset = subset
        self.shuffle = shuffle
        self.info = info
        self.width = width
        self.height = height
        self.file_col = file_col
        self.augmentation_function = augmentation_function
        self.preprocess_image = preprocess_image
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        
        # Create tha X empty structure
        X = np.empty(
            (2 * self.batch_size, 1, self.height, self.width, 3),
            dtype=np.float32,
        )

        # get data to use
        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        # Add shuffle in order to avoid network recalling fixed order
        shuffle_a = np.arange(self.batch_size)
        shuffle_b = np.arange(self.batch_size)

        # Random input order for training to avoid the network to overfit to the positions
        if self.subset == "train":
            random.shuffle(shuffle_a)
            random.shuffle(shuffle_b)

        # Create labels empty structures
        labels_ab_aa = np.zeros((self.batch_size, 2 * self.batch_size))
        labels_ba_bb = np.zeros((self.batch_size, 2 * self.batch_size))

        for i, filename in enumerate(self.df.iloc[indexes][self.file_col]):
            # Load image
            self.info[index * self.batch_size + i] = filename
            img = img_to_array(load_img(filename))
            
            # Make two different augmentations of the same image
            img_T1 = self.augmentation_function(
                img,
                self.height,
                self.width,
                is_training=self.subset=='train'
            )
            img_T2 = self.augmentation_function(
                img,
                self.height,
                self.width,
                is_training=self.subset=='train'
            )
            
            # Preprocess image for the neural network
            img_T1 = self.preprocess_image(img_T1)
            img_T2 = self.preprocess_image(img_T2)

            # T1-images between 0 -> batch_size - 1
            X[shuffle_a[i]] = img_T1
            # T2-images between batch_size -> 2*batch_size - 1
            X[self.batch_size + shuffle_b[i]] = img_T2

            # label ab
            labels_ab_aa[shuffle_a[i], shuffle_b[i]] = 1
            # label ba
            labels_ba_bb[shuffle_b[i], shuffle_a[i]] = 1

        y = tf.concat([labels_ab_aa, labels_ba_bb], 1)

        return list(X), y
