import tensorflow as tf

def use_gpu_and_allow_growth():
    '''Prints the list of availables GPUs and set the allow memory grow in all of them.
    
    By checking the available GPUs tensorflow automatically uses the GPUs.
    '''
    
    print(f'Num GPUs Available: {len(tf.config.experimental.list_physical_devices("GPU"))}')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)