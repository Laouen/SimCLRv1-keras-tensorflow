import argparse
import os
import glob
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

from SimCLR_Keras.preprocessing import preprocess_image, AUGMENTATION_FUNCTIONS

def compile_script_arguments():
    parser = argparse.ArgumentParser(
        description='Train SimCLR constrastive model with the VGG16 as the base model.'
    )

    parser.add_argument(
        'save_path',
        type=str,
        help='The path where to save the resulting trained weights'
    )
    
    parser.add_argument(
        'data_dir',
        type=str,
        help='The path of the train/val data with a column named filename and images'
    )
    
    parser.add_argument(
        '--df_sep',
        type=str,
        default='|',
        help='The separator character to use for pandas sep argument. Go to pandas.read_csv documentation for more information'
    )
    
    parser.add_argument(
        '--input_shape',
        type=int,
        nargs=3,
        default=[80,80,3],
        help='The shape of the input image as (with, height, channels).'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The processing batch size.'
    )
    
    parser.add_argument(
        '--feat_dims_ph',
        type=int,
        nargs='+',
        default=[2048, 128],
        help='The dimensions of the projection head layers.'
    )
    parser.add_argument(
        '--augmentation_functions',
        type=str,
        nargs='+',
        choices=AUGMENTATION_FUNCTIONS,
        default=['crop', 'color_distort'],
        help='The augmentation function to apply to the images in the order of application to generate positive pairs.'
    )
    
    parser.add_argument(
        '--num_of_unfrozen_layers',
        type=int,
        help='The number of layers to pretrain from the base model in top-down order.'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='The number of epochs to strain a model. Default is 500'
    )
    
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='The patience parameter for the early stop callback. Default is 10'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.25,
        help='The proportion of the data to use for evaluate the model. Default is 0.25'
    )
    
    parser.add_argument(
        '--use_gpu',
        default=False,
        action='store_true',
        help='Use GPU to train.'
    )

    return parser.parse_args()

def main(save_path,
         data_dir,
         df_sep='|',
         input_shape=(80,80,3),
         batch_size=32,
         feat_dims_ph=[2048, 128],
         augmentation_functions=['crop', 'color_distort'],
         num_of_unfrozen_layers=1,
         epochs=1000, 
         patience=10, 
         test_size=0.25,
         use_gpu=True):
    
    if use_gpu:
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
    initial_epoch = 0
    checkpoint_files_pattern = os.path.join(save_path, f'checkpoints/*_checkpoint.h5')
    all_checkpoint_files = glob.glob(checkpoint_files_pattern)
    if len(all_checkpoint_files) > 0:
        initial_epoch = max([
            int(checkpoint_file.split('/')[-1].split('_')[0]) 
            for checkpoint_file in all_checkpoint_files
        ])
        checkpoint_file = os.path.join(save_path, f'checkpoints/{initial_epoch}_checkpoint.h5')
        print(f'Usin checkpoint file {checkpoint_file}')
        model.load_weights(checkpoint_file)
        time.sleep(4)
            
    
    print('\n*** Plot model ***')
    model.plot_model()

    print(f'\n*** Build Data generators for train/val using augmentations {augmentation_functions}***')
    image_augmentation = partial(
        preprocess_image,
        operators=augmentation_functions
    )
    params_generator = {
        'batch_size': batch_size,
        'shuffle' : True,
        'width': input_shape[0],
        'height': input_shape[1],
        'augmentation_function': image_augmentation,
        'preprocess_image': preprocess_input
    }


    df = pd.read_csv(os.path.join(data_dir,'output_tags.csv'), sep=df_sep)
    df['filename'] = df.id.apply(lambda x: os.path.join(data_dir, f'images/{x}.jpg'))
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=42 # This ensures splits will be equal en all runs
    )

    print(f'Total rows in dataframe {len(df)}')
    print(f'Total rows in test {len(df_test)}')
    print(f'Total rows in train {len(df_train)}')

    # Generators
    data_train = DataGeneratorSimCLR(
        df_train.reset_index(drop=True),
        subset='train',
        **params_generator
    )

    data_val = DataGeneratorSimCLR(
        df_test.reset_index(drop=True),
        subset='val',
        **params_generator
    )

    print('\n*** Predict on validation before ***')
    y_predict_val_before = model.predict(data_val)
    print(f'Accuracy - test - before: {np.round(np.sum(data_val[0][1] * y_predict_val_before[:batch_size])/(2*batch_size),2)}')
    
    print(f'\n*** Train SimCLR model from epoch {initial_epoch}')
    model.train(
        data_train,
        data_val,
        epochs=epochs,
        initial_epoch=initial_epoch,
        patience=patience
    )
    
    print('Remove checkpoint files')
    [os.remove(checkpoint_file) for checkpoint_file in glob.glob(checkpoint_files_pattern)]

    print('\n*** Predict on validation after and final results ***')
    y_predict_test_after = model.predict(data_val)
    
    print(f'Random guess accuracy: {round(1 / (2*batch_size), 4)}')
    print(f'Accuracy - test - before: {np.round(np.sum(data_val[0][1] * y_predict_val_before[:batch_size])/(2*batch_size),2)}')
    print(f'Accuracy - test - after: {np.round(np.sum(data_val[0][1] * y_predict_test_after[:batch_size])/(2*batch_size),2)}')

    
if __name__ == '__main__':

    args = compile_script_arguments()

    main(**vars(args))