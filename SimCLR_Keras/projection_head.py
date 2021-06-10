from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l1


def build_projection_head(
    feat_dims, 
    input_shape=(None, 2048),
    activation='relu',
    regul=0.005,
    name='Projection_head'
):
    model = Sequential(name=name)
    for i,size in enumerate(feat_dims[:-1]):
        model.add(
            Dense(
                size,
                activation=activation,
                kernel_regularizer=l1(regul),
                name=f'{name}_{i}'
            )
        )
    model.add(
        Dense(
            feat_dims[-1],
            kernel_regularizer=l1(regul),
            name=f'{name}_{len(feat_dims)}'
        )
    )
    model.build(input_shape)
    return model