import argparse

from ml_models.ml_model_base import MLModel
from ml_models.mlp import MLP


class MlModelFactory:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args


    def get_model(self, model_name: str) -> MLModel:
        match model_name:
            case 'mlp':
                return MLP(self.args)
            case _:
                print('No model called {model}. Please select a valid model in config.yaml.'.format(
                    model=model_name,
                ))
                raise NotImplementedError