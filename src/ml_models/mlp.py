import pandas
import argparse
import tensorflow

from ml_models.ml_model_base import MLModel


class MLP(MLModel):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)

        callback_list = [
            tensorflow.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                verbose=1,
            ),
            tensorflow.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=5,
                min_lr=0.001
            ),
        ]

        self.callbacks.extend(callback_list)


    def get_model(self) -> tensorflow.keras.Model:
        # JUST A LITTLE MLP
        pass


    def train(self) -> tensorflow.keras.callbacks.History:
        print('Training MLP...')
        data_dict = self.training_data
        assert(len(data_dict) > 0)

        model = self.get_model()
        
        model.summary()
        
        if self.args.plot_model:
            tensorflow.keras.utils.plot_model(
                model=model, 
                to_file=self.exp_directory + self.args.cas + '_training_model.png',
                show_shapes=True,
            )

        model.compile(
            loss=tensorflow.keras.losses.MeanSquaredError(),
            optimizer=tensorflow.keras.optimizers.Adam(self.args.mlp_adam_lr),
        )

        history = model.fit(
            x=data_dict['train_x'],
            y=data_dict['train_y'],
            batch_size=self.args.mlp_batch_size,
            epochs=self.args.mlp_epochs,
            shuffle=True,
            validation_data=(data_dict['valid_x'], data_dict['valid_y']),
            callbacks=self.callbacks,
        )

        print('Saving trained model to {path}'.format(path=self.train_model_path))
        model.save(self.train_model_path)
        print('Saving trained weights to {path}'.format(path=self.train_weights_path))
        model.save_weights(self.train_weights_path)

        print('Done training the MLP.')
        return history


    def inference(self) -> pandas.DataFrame:
        data_dict = self.inference_data
        assert(len(data_dict) > 0)

        model = tensorflow.keras.models.load_model(self.train_model_path)

        model.summary()
        
        if self.args.plot_model:
            tensorflow.keras.utils.plot_model(
                model=model, 
                to_file=self.exp_directory + self.args.cas + '_mlp_inference_model.png',
                show_shapes=True,
            )

        print('Inference with trained weights {path}'.format(path=self.train_weights_path))
        model.load_weights(self.train_weights_path)

        pred_y = model.predict(data_dict['test_x']).flatten()

        output_dataframe = pandas.DataFrame({
            'placeholder': data_dict['placeholder'],
            'predicted_label': pred_y,
        })

        output_dataframe.to_csv(self.inference_output_path, index=False)
        print('Saved predictions to {path}'.format(path=self.inference_output_path))
        
        return output_dataframe
