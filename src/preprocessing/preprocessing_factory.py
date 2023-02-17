import argparse

from preprocessing.preprocessing_base import PreprocessingBase
from preprocessing.mlp_preprocessing import MLPPreprocessing


class PreProcessingFactory:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args


    def get_preprocessor(self, model_name: str) -> PreprocessingBase:
        match model_name:
            case 'mlp':
                return MLPPreprocessing(self.args)
            case _:
                print('No model called {model}. Please select a valid model in config.yaml.'.format(
                    model=model_name,
                ))
                raise NotImplementedError