import numpy as np
from tensorflow.keras.utils import Sequence

class StockSequence(Sequence):
    """
    A sequence class for generating batches of stock market data for training a machine learning model.

    Attributes:
        prices (array-like): Sequence of stock prices.
        volumes (array-like): Sequence of stock trading volumes.
        ETF_prices (array-like): Sequence of ETF prices.
        batch_size (int): Number of sequences per batch.
        sequence_length (int): Number of time steps in each sequence.

    The class is designed to be used with Keras models that require batched input.
    """

    def __init__(self, prices, volumes, ETF_prices, batch_size, sequence_length):
        """
        Initializes the StockSequence with stock market data and batch configuration.

        Parameters:
            prices (array-like): Sequence of stock prices.
            volumes (array-like): Sequence of stock trading volumes.
            ETF_prices (array-like): Sequence of ETF prices.
            batch_size (int): Number of sequences per batch.
            sequence_length (int): Number of time steps in each sequence.
        """
        self.prices = prices
        self.volumes = volumes
        self.ETF_prices = ETF_prices
        self.batch_size = batch_size
        self.sequence_length = sequence_length

    def __len__(self):
        """
        Determines the number of batches in the sequence.

        Returns:
            int: The total number of batches in the sequence.
        """
        return len(self.prices) - self.sequence_length

    def __getitem__(self, index):
        """
        Retrieves a batch at the given index.

        Parameters:
            index (int): Index of the batch to retrieve.

        Returns:
            dict: A dictionary with two keys - 'sequence_with_features', containing
                  the input features for the model, and 'target', containing the target
                  stock price.
        """
        start_index = index
        end_index = index + self.sequence_length
        price_sequence = self.prices[start_index:end_index]
        volume_sequence = self.volumes[start_index:end_index]
        ETF_price_sequence = self.ETF_prices[start_index:end_index]

        sequence_with_features = np.column_stack((price_sequence, volume_sequence, ETF_price_sequence))
        price_target = self.prices[end_index]

        return {"sequence_with_features": sequence_with_features, "target": price_target}
