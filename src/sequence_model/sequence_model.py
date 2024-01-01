from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np

class SequenceModel:
    def __init__(self, training_data):
        
        self.training_data = training_data

        #Extract number of features from entered data
        series_first_sequence = training_data[0]["sequence_with_features"] #first index: sequence index, second index: "sequence_with_features" (one for each feature) or "target" 
        self.sequence_length = series_first_sequence.shape[0]
        #series_first_sequence is 2D array with shape (sequence_length, nbr_features)
        self.nbr_features = series_first_sequence.shape[1] 
        self.model = None

    def compile_model(self):
        # Define the model architecture
        model = Sequential()

        # for LSTM:
        model.add(LSTM(64, activation="relu", input_shape=(self.sequence_length,self.nbr_features), return_sequences=True))
        model.add(LSTM(64, activation="relu", return_sequences=False))

        model.add(Dense(32, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))  # Output layer with 1 neuron for regression

        # Compile the model
        model.compile(loss="mean_squared_error", optimizer="adam")

        self.model = model

        return self
    
    def fit(self, epochs: int, batch_size: int):
        self.epochs = epochs
        self.batch_size = batch_size
        input_data = self._get_sequences(data_raw=self.training_data)
        target_data = self._get_targets(data_raw=self.training_data)
        self.model.fit(input_data, target_data, epochs=self.epochs, batch_size=self.batch_size)

    def predict(self, input_data):
        sequences = self._get_sequences(input_data)
        predictions = self.model.predict(sequences)

        return predictions

    @staticmethod
    def _get_sequences(data_raw):
        return np.array([data["sequence_with_features"] for data in data_raw])

    @staticmethod
    def _get_targets(data_raw):
        return np.array([data["target"] for data in data_raw])
    