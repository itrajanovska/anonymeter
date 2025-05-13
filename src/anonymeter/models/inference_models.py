from autogluon.tabular import TabularPredictor
import pandas as pd


class NNModel:
    def __init__(self, train_data, label):
        self.predictor = TabularPredictor(label=label)
        self.train_data = train_data

    def fit(self) -> TabularPredictor:
        self.predictor.fit(
            train_data=self.train_data,
            hyperparameters={
                'NN_TORCH': {},  # Include neural network (Torch-based)
            }
        )
        return self.predictor

    def predict(self, x):
        return self.predictor.predict(x)
