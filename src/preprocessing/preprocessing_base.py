import argparse
from abc import ABC, abstractmethod


class PreprocessingBase(ABC):
    def __init__(self, args: argparse.Namespace) -> None:
        pass


    @abstractmethod
    def preprocess_train(self) -> dict:
        '''
        Loads the input file. Shuffles the dataset and splits into train/test. 
        
        ## Returns:
        dict({
            'train_x': train_x,
            'train_y': train_y,
            'valid_x': valid_x,
            'valid_y': valid_y,
        })

        Where values are returned by sklearn.model_selection.train_test_split
        '''


    @abstractmethod
    def preprocess_inference(self) -> dict:
        '''
        Reads the inference input file (inference_csv in config.yaml).
        
        return dict({
            'placeholder': test_x_raw,
            'test_x': test_x,
        })
        '''
