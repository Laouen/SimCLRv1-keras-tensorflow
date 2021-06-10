import os
from pathlib import Path
from datetime import datetime

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from SimCLR_Keras.layers import SoftmaxCosineSim
from SimCLR_Keras.classifier import Classifier
from SimCLR_Keras.activations import swish
from SimCLR_Keras.projection_head import build_projection_head

from VincentVGG.Callbacks import WeightsChangeTracker, RestartTrainingModelCheckpoint

import numpy as np


class SimCLR:
    """
    SimCLR-class contains among others a SimCLR keras-model
    The SimCLR_model has
        - (2 * batch_size) inputs of shape = (feat_dims_ph[-1])
        - base_model which is stored independently to evaluate its feature quality
        - flatten_layer
        - projection_head
        - 1 output = matrix of shape (batch_size x 4.batch_size)
    """

    def __init__(self,
                 base_model,
                 input_shape,
                 batch_size,
                 feat_dims_ph,
                 num_of_unfrozen_layers=None,
                 ph_activation='relu',
                 ph_regul=0.005,
                 lr=1e-4,
                 loss="categorical_crossentropy",
                 save_path="models/trashnet",
                 r=1):
        
        self.base_model = base_model
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.feat_dims_ph = feat_dims_ph
        self.num_of_unfrozen_layers = num_of_unfrozen_layers
        self.ph_activation = ph_activation
        self.ph_regul = ph_regul
        self.lr = lr
        self.optimizer = Adam(lr, amsgrad=True)
        self.loss = loss
        self.save_path = save_path
        self.r = r
        
        # Create save directory
        Path(self.save_path).mkdir(parents=True, exist_ok=True)

        # Different layers around base_model
        self.flatten_layer = Flatten()
        self.soft_cos_sim = SoftmaxCosineSim(
            batch_size=self.batch_size,
            feat_dim=self.feat_dims_ph[-1]
        )
        
        # Projection head
        self.ph_l = build_projection_head(
            feat_dims_ph,
            input_shape=Flatten()(self.base_model.output).shape,
            activation=self.ph_activation,
            regul=self.ph_regul,
            name='Projection_head'
        )

        self.SimCLR_model = self.build_model()

    def build_model(self):
        """ Building SimCLR_model
        """

        if self.num_of_unfrozen_layers is not None:
            # Set trainable only the last num_of_unfrozen_layers 
            for layer in self.base_model.layers[: -self.num_of_unfrozen_layers]:
                layer.trainable = False
            for layer in self.base_model.layers[-self.num_of_unfrozen_layers :]:
                layer.trainable = True
        else:
            # Set all base model layers as trainable
            for layer in self.base_model.layers:
                layer.trainable = True
                
        self.i = []  # Inputs (# = 2 x batch_size)
        self.f_x = []  # Output base_model
        self.h = []  # Flattened feature representation
        self.g = []  # Projection head

        # Getting learnable building blocks
        for index in range(2 * self.batch_size):
            self.i.append(Input(shape=self.input_shape))
            self.f_x.append(self.base_model(self.i[index]))
            self.h.append(self.flatten_layer(self.f_x[index]))
            self.g.append(self.ph_l(self.h[index]))

        self.o = self.soft_cos_sim(self.g)  # Output = Last layer of projection head

        # Combine model and compile
        SimCLR_model = Model(inputs=self.i, outputs=self.o)
        SimCLR_model.compile(optimizer=self.optimizer, loss=self.loss)
        return SimCLR_model

    def train(self, data_train, data_val, epochs=10, initial_epoch=0, patience=10, pr=True):
        """ Training the SimCLR model and saving best model with time stamp
            Transfers adapted weights to base_model
        """

        # Fit
        self.SimCLR_model.fit(
            data_train,
            epochs=epochs,
            initial_epoch=initial_epoch,
            verbose=1,
            validation_data=data_val,
            callbacks=self.get_callbacks(patience),
        )

        # Print number of trainable weights
        if pr:
            self.print_weights()

        # Save
        self.save_base_model_weights()
        self.save_projection_head_weights()

    def unfreeze_and_train(
        self,
        data_train,
        data_val,
        num_of_unfrozen_layers,
        r,
        lr=1e-4,
        epochs=10,
        patience=10,
        pr=True):
        """ Changes number of unfrozen layers in the base model and rebuilds it
            Training the SimCLR model and saving best model with time stamp
            Transfers adapted weights to base_model
        """
        # Update parameters
        self.num_of_unfrozen_layers = num_of_unfrozen_layers
        self.r = r
        if self.lr != lr:
            self.change_lr(lr)

        # (Un)freeze layers of base_model
        self.SimCLR_model = self.build_model()

        # Print number of trainable weights
        if pr:
            self.print_weights()

        # Train
        self.train(data_train, data_val, epochs, patience)

    def predict(self, data):
        """ SimCLR prediction
        """
        return self.SimCLR_model.predict(data)

    def save_base_model(self, path=None):
        """ Save base_model with time stamp
        """
        
        if path is None:
            file_dir = os.path.join(self.save_path, "saved_models")
            Path(file_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(file_dir, f'base_model_round_{self.r}.h5')

        self.base_model.save(path)

    def save_base_model_weights(self, path=None):
        """ Save base_model with time stamp
        """
        
        if path is None:
            file_dir = os.path.join(self.save_path, "saved_models")
            Path(file_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(file_dir, f'base_model_round_{self.r}_weights.h5')

        self.base_model.save_weights(path)
    
    def save_projection_head_weights(self, path=None):
        """ Save base_model with time stamp
        """
        
        if path is None:
            file_dir = os.path.join(self.save_path, "saved_models")
            Path(file_dir).mkdir(parents=True, exist_ok=True)
            path = os.path.join(file_dir, f'projection_head_round_{self.r}_weights.h5')

        self.ph_l.save_weights(path)
    
    def load_weights(self, path):
        self.SimCLR_model.load_weights(path)

    def change_lr(self, lr):
        """ Changing learning rate of SimCLR_model
        """
        self.lr = lr
        K.set_value(self.SimCLR_model.optimizer.learning_rate, self.lr)

    def get_callbacks(self, early_stop_patience=10):
        """ Returns callbacks used while training
        """

        # Tensorboard callback
        # Time stamp for checkpoint
        dt_string = datetime.now().strftime("_%m_%d_%Hh_%M")
        tensorboard_dir = os.path.join(self.save_path, f'tensorboard/logs/fit{dt_string}_training')
        Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
        tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)

        # checkpoint save last epoch model to resume training
        Path(os.path.join(self.save_path, f'checkpoints')).mkdir(parents=True, exist_ok=True)
        training_resume_checkpoint = RestartTrainingModelCheckpoint(
            info_file_path=os.path.join(self.save_path, f'checkpoints/last_checkpoint_info.npy'),
            training_number=0,
            filepath=os.path.join(self.save_path, f'checkpoints/last_checkpoint.h5'),
            monitor='val_loss',
            mode='min',
            verbose=0,
            save_best_only=False,
            save_weights_only=False
        )
        
        # Checkpoint to save best model weights
        best_checkpoint_dir = os.path.join(self.save_path,'results') 
        Path(best_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        best_checkpoint = ModelCheckpoint(
            os.path.join(best_checkpoint_dir, f'{dt_string}_best_checkpoint.h5'),
            verbose=1,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
        )
    
        # Early stop if model stop improving
        earlyStopping = EarlyStopping(
            verbose=0,
            monitor="val_loss",
            mode="min",
            patience=early_stop_patience,
            restore_best_weights=True
        )

        # Learning rate scheduler
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            verbose=1,
            factor=0.5
        )
        
        # Track base mode weights change
        base_model_weights_change = WeightsChangeTracker(self.base_model, 'VGG16')
        ph_weights_change = WeightsChangeTracker(self.ph_l, 'Projection head')

        return [
            tensorboard,
            training_resume_checkpoint,
            best_checkpoint,
            earlyStopping,
            reduce_lr,
            base_model_weights_change,
            ph_weights_change
        ]

    def plot_model(self, filename='model_architecture.png'):
        plot_model(
            self.SimCLR_model,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            to_file=os.path.join(self.save_path,filename)
        )
    
    def print_weights(self):
        """ Function to print (non)-learnable weights
            Helps checking unfreezing process
        """
        trainable_count = np.sum([
            K.count_params(w) for w in self.SimCLR_model.trainable_weights
        ])

        non_trainable_count = np.sum([
            K.count_params(w) for w in self.SimCLR_model.non_trainable_weights
        ])

        print(f"Trainable parameters: {round(trainable_count/1e6,2)} M.")
        print(f"Non-trainable parameters: {round(non_trainable_count/1e6,2)} M.")

    def train_NL_and_evaluate(
        self,
        dfs,
        batch_size,
        params_generator,
        fraction,
        class_labels,
        reg_dense=0.005,
        reg_out=0.005,
        nums_of_unfrozen_layers=[5, 5, 6, 7],
        lrs=[1e-3, 1e-4, 5e-5, 1e-5],
        epochs=[5, 5, 20, 25],
        verbose_epoch=0,
        verbose_cycle=1):
        """ Trains and evaluates a nonlinear classifier on top of the base_model
        """
        results = {"acc": 0}
        for i in range(5):
            if verbose_cycle:
                print(f"Learning attempt {i+1}")

            classifier = Classifier(
                base_model=self.base_model,
                num_classes=params_generator["num_classes"],
                reg_dense=reg_dense,
                reg_out=reg_out,
            )

            data_train, data_val, data_test = classifier.get_generators(
                dfs, fraction, batch_size, params_generator
            )

            classifier.train(
                data_train,
                data_val,
                fraction,
                nums_of_unfrozen_layers,
                lrs,
                epochs,
                verbose_epoch,
                verbose_cycle,
            )
            acc, report = classifier.evaluate_on_test(
                dfs["test"], data_test, class_labels
            )

            if results["acc"] < acc:
                results["acc"] = acc
                results["report"] = report
                results["attempt"] = i + 1

        print("Best result from attempt", str(results["attempt"]))
        print(results["report"])
