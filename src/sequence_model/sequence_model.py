from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np


class SequenceModel:
    """
    A class for creating and training a sequential model for time series prediction.

    This class builds a neural network model using LSTM layers followed by Dense layers.

    Attributes:
        training_data (list): A list of training data, where each element is a dictionary 
                              with keys "sequence_with_features" and "target".
        sequence_length (int): The length of each input sequence.
        nbr_features (int): The number of features in each sequence.
        model (keras.Model): The compiled Keras sequential model.
    """

    def __init__(self, training_data):
        """
        Initializes the SequenceModel with given training data.

        Args:
            training_data (list): A list of dictionaries, each containing a sequence 
                                  of features and a target value.
        """
        self.training_data = training_data
        series_first_sequence = training_data[0]["sequence_with_features"]
        self.sequence_length = series_first_sequence.shape[0]
        self.nbr_features = series_first_sequence.shape[1]
        self.model = None

    def compile_model(self):
        """
        Compiles the LSTM-based sequential model with multiple Dense layers.

        The model architecture includes two LSTM layers and three Dense layers.

        Returns:
            self (SequenceModel): The instance of the model with the compiled neural network.
        """
        model = Sequential()
        model.add(LSTM(64, activation="relu", input_shape=(self.sequence_length, self.nbr_features), return_sequences=True))
        model.add(LSTM(64, activation="relu", return_sequences=False))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))  # Output layer with 1 neuron for regression
        model.compile(loss="mean_squared_error", optimizer="adam")
        self.model = model
        return self

    def fit(self, epochs: int, batch_size: int):
        """
        Trains the compiled model on the training data.

        Args:
            epochs (int): The number of epochs to train the model.
            batch_size (int): The size of the batches of data to use in training.

        Trains the model using internally extracted sequences and targets from the training data.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        input_data = self._get_sequences(data_raw=self.training_data)
        target_data = self._get_targets(data_raw=self.training_data)
        self.model.fit(input_data, target_data, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, input_data):
        """
        Makes predictions using the trained model on new data.

        Args:
            input_data (list): The new data on which to make predictions, structured 
                               similarly to training_data.

        Returns:
            numpy.ndarray: The predictions made by the model.
        """
        sequences = self._get_sequences(input_data)
        predictions = self.model.predict(sequences)
        return predictions

    @staticmethod
    def _get_sequences(data_raw):
        """
        Extracts the sequences of features from raw data.

        Args:
            data_raw (list): The raw data from which to extract sequences.

        Returns:
            numpy.ndarray: An array of extracted sequences.
        """
        return np.array([data["sequence_with_features"] for data in data_raw])

    @staticmethod
    def _get_targets(data_raw):
        """
        Extracts the target values from raw data.

        Args:
            data_raw (list): The raw data from which to extract target values.

        Returns:
            numpy.ndarray: An array of target values.
        """
        return np.array([data["target"] for data in data_raw])
