from SimCLR_Keras.Scripts.pretain_vgg16 import pretain_vgg16
import argparse

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

if __name__ == '__main__':

    args = compile_script_arguments()
    pretain_vgg16(**vars(args))