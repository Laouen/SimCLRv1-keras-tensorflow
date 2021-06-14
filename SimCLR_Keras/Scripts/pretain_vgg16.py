import os
import time
from functools import partial

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from SimCLR_Keras.DataGenerators.DataGeneratorSimCLR import DataGeneratorSimCLR
from SimCLR_Keras.model import SimCLR
from SimCLR_Keras.gpu import use_gpu_and_allow_growth
from SimCLR_Keras.preprocessing import preprocess_image

from VincentVGG.Utils import retrieve_training_state, remove_training_state

def pretain_vgg16(
    save_path,
    data_dir,
    df_sep='|',
    input_shape=(80, 80, 3),
    batch_size=32,
    feat_dims_ph=[2048, 128],
    augmentation_functions=['crop', 'color_distort'],
    num_of_unfrozen_layers=1,
    epochs=1000,
    patience=10,
    test_size=0.25,
    gpu_allow_growth=False,
    random_state=None): # Set a fixed value to ensure same test_train data split over different runs
    """Train SimCLR constrastive model with the VGG16 as the base model.

    Parameters:
    save_path (str): The path where to save the resulting trained weights
    data_dir (str): The path of the train/val data with a column named filename and images.
    df_sep (str): The separator character to use for pandas sep argument. Go to pandas.read_csv documentation for more information. Default is '|'.
    input_shape (tuple(int)): The shape of the input image as (with, height, channels). Default is (80, 80, 3).
    batch_size (int): The processing batch size. Default is 32.
    feat_dims_ph (array(int)): The dimensions of the projection head layers. Default is [2048, 128].
    augmentation_functions (array(str)): The augmentation function to apply to the images in the order of application to generate positive pairs. Default is ['crop', 'color_distort'].
    num_of_unfrozen_layers (int): The number of layers to pretrain from the base model in top-down order. Default is 1.
    epochs (int): Maximum number of epochs to train de model. Default is 1000.
    patience (int): Maximum number of consecutive epochs without improvement to allow before early stop. Default is 10.
    test_size (float): The proportion of the data to use for evaluate the model. Default is 0.25.
    use_gpu (boolean): If true, the GPU allow growth memory is set. This is needed to detect the GPU in some computers. Default is False.
    """

    if gpu_allow_growth:
        print('Initialize GPU', flush=True)
        use_gpu_and_allow_growth()

    # base model
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    print('\n*** Build VGG16 base model loaded ***')
    base_model.summary()

    print('\n*** Build SimCLR model to train ***')
    model = SimCLR(
        base_model=base_model,
        input_shape=input_shape,
        batch_size=batch_size,
        feat_dims_ph=feat_dims_ph,
        num_of_unfrozen_layers=num_of_unfrozen_layers,
        save_path=save_path,
        lr=1e-5
    )

    # Check and retrieve checkpoint file to resume training
    _, initial_epoch, checkpoint_file = retrieve_training_state(save_path)  

    # Check and retrieve checkpoint file to resume training
    print('\n========= Restore checkpoint =========')
    print(f'Usin checkpoint file {checkpoint_file}')
    print(f'Initial epcho is {initial_epoch}')
    time.sleep(4)

    print('\n========= Build model =========')
    model.build_model(checkpoint_file)

    print('\n========= Plot model =========')
    model.plot_model()

    print(f'\n========= Build Data generators for train/val using augmentations {augmentation_functions} =========')
    image_augmentation = partial(
        preprocess_image,
        operators=augmentation_functions
    )
    generator_params = {
        'batch_size': batch_size,
        'shuffle': True,
        'width': input_shape[0],
        'height': input_shape[1],
        'augmentation_function': image_augmentation,
        'preprocess_image': preprocess_input
    }

    df = pd.read_csv(
        os.path.join(data_dir, 'output_tags.csv'),
        sep=df_sep
    )

    df['filename'] = df.id.apply(lambda x: os.path.join(data_dir, f'images/{x}.jpg'))
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=random_state
    )

    print(f'Total rows in dataframe {len(df)}')
    print(f'Total rows in test {len(df_test)}')
    print(f'Total rows in train {len(df_train)}')

    # Generators
    data_train = DataGeneratorSimCLR(
        df_train.reset_index(drop=True),
        subset='train',
        **generator_params
    )

    data_val = DataGeneratorSimCLR(
        df_test.reset_index(drop=True),
        subset='val',
        **generator_params
    )

    print('\n========= Predict on validation before =========')
    y_predict_val_before = model.predict(data_val)
    accuracy = np.sum(data_val[:][1] * y_predict_val_before)/(2*batch_size)

    print(f'Accuracy - test - before: {np.round(accuracy,2)}')


    print(f'\n========= Train SimCLR model from epoch {initial_epoch} =========')
    model.train(
        data_train,
        data_val,
        epochs=epochs,
        initial_epoch=initial_epoch,
        patience=patience
    )

    print('\n========= Remove checkpoint files =========')
    remove_training_state(save_path)

    print('\n========= Predict on validation after and final results =========')
    y_predict_test_after = model.predict(data_val)

    print(f'Random guess accuracy: {round(1 / (2*batch_size), 4)}')
    print(
        f'Accuracy - test - before: {np.round(np.sum(data_val[0][1] * y_predict_val_before[:batch_size])/(2*batch_size),2)}')
    print(
        f'Accuracy - test - after: {np.round(np.sum(data_val[0][1] * y_predict_test_after[:batch_size])/(2*batch_size),2)}')
        
