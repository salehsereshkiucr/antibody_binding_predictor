import os
import numpy
import pandas
import argparse
from sklearn.model_selection import train_test_split

from preprocessing.preprocessing_base import PreprocessingBase


class MLPPreprocessing(PreprocessingBase):
    def __init__(self, args):
        self.args = args


    def preprocess_train(self) -> dict:
        input_path = os.path.join(self.args.input_directory, self.args.train_csv)
        
        print('Loading training data from {path}'.format(path=input_path))
        train_df = pandas.read_csv(input_path).sample(frac=1)  # Shuffles the dataset.
        print('Done.')

        X = train_df[self.args.train_guide_seq_col_name].to_list()
        Y = train_df[self.args.train_guide_score_col_name].to_list()

        Y = numpy.asarray(Y).reshape((-1, 1))  # column vector

        train_x, valid_x, train_y, valid_y = train_test_split(
            X,
            Y,
            train_size=self.args.train_test_ratio,
        )

        print('Training on {train_n} and validating on {valid_n} examples. Ratio: {ratio}.'.format(
            train_n=len(train_x),
            valid_n=len(valid_x),
            ratio=self.args.train_test_ratio,
        ))

        return dict({
            'train_x': train_x,
            'train_y': train_y,
            'valid_x': valid_x,
            'valid_y': valid_y,
        })


    def preprocess_inference(self) -> dict:
        path = os.path.join(self.args.input_directory, self.args.inference_csv)
        df = pandas.read_csv(path)

        test_x_raw = df[self.args.inference_col_name].to_list()
        # test_x = self.encode_guides(test_x_raw)

        return dict({
            # 'test_x': test_x,
            'placeholder': test_x_raw,
        })