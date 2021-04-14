import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.utils import data_utils


class DataGeneratorClass(data_utils.Sequence):
    def __init__(
        self,
        df,
        batch_size=16,
        subset="train",
        shuffle=True,
        info={},
        max_width=80,
        max_height=80,
        num_classes=5,
        preprocess_input=lambda x: x,
        augmentation=None,
        file_col='filename'
    ):
        super().__init__()
        self.df = df
        self.indexes = np.asarray(self.df.index)
        self.shuffle = shuffle
        self.subset = subset
        self.batch_size = batch_size
        self.info = info
        self.max_width = max_width
        self.max_height = max_height
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.preprocess_input = preprocess_input
        self.file_col = file_col
        self.datagen = self.datagen()
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def datagen(self):
        return ImageDataGenerator()

    def __getitem__(self, index):
        X = np.empty(
            (self.batch_size, self.max_height, self.max_width, 3),
            dtype=np.float32,
        )
        y = np.empty((self.batch_size, self.num_classes), dtype=np.float32,)

        indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]

        for i, row in enumerate(self.df.iloc[indexes].iterrows()):
            filename = row[1][self.file_col]
            self.info[index * self.batch_size + i] = filename

            X[i,] = self.preprocess_input(
                img_to_array(load_img(filename))
            )

            if self.subset == "train":
                y[i,] = row[1]["class_one_hot"]

        # [None] is used to silence warning
        # https://stackoverflow.com/questions/59317919/warningtensorflowsample-weight-modes-were-coerced-from-to
        return X, y
