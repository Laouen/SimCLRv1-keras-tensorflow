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
    def __init__(self,
                 df,
                 batch_size=16,
                 subset="train",
                 shuffle=True,
                 info={},
                 width=80,
                 height=80,
                 file_col='filename',
                 augmentation_function=default_augmentation,
                 preprocess_image=lambda x: x):

        """

        Note: If the batch_size is not a multiple of the df lenght, then the las entries of df are discarded as we need all batches to be the same size.
        """

        super().__init__()
        self.df = df
        self.data_indexes = np.asarray(self.df.index)
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
        # Data generator drops last images from self.df if they don't complete a full batch
        return len(self.df) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data_indexes)

    def __getitem__(self, batch_index):
        
        # Create tha X empty structure (final shape will be (2*batch_size, 1, height, width, 3))
        X = [None for i in np.arange(2 * self.batch_size)]

        # get data to use
        indexes_to_use = self.data_indexes[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]

        # Add an index redirection map to allow shuffle in order to avoid network recalling fixed order if training
        index_map_a = np.arange(self.batch_size)
        index_map_b = np.arange(self.batch_size)

        # Random input order for training to avoid the network to overfit to the positions
        if self.subset == "train":
            random.shuffle(index_map_a)
            random.shuffle(index_map_b)

        # Create labels empty structures
        labels_ab_aa = np.zeros((self.batch_size, 2 * self.batch_size))
        labels_ba_bb = np.zeros((self.batch_size, 2 * self.batch_size))

        for i, filename in enumerate(self.df.loc[indexes_to_use][self.file_col]):
            # Load image
            self.info[batch_index * self.batch_size + i] = filename
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
            X[index_map_a[i]] = tf.reshape(img_T1, (1, self.height, self.width, 3))
            # T2-images between batch_size -> 2*batch_size - 1
            X[self.batch_size + index_map_b[i]] = tf.reshape(img_T2, (1, self.height, self.width, 3))

            # label ab
            labels_ab_aa[index_map_a[i], index_map_b[i]] = 1
            # label ba
            labels_ba_bb[index_map_b[i], index_map_a[i]] = 1

        y = tf.concat([labels_ab_aa, labels_ba_bb], 1)

        return X, y
