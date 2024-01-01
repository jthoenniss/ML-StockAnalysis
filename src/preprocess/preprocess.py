from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


class StockPricePreprocessor:
    """
    A class for preprocessing stock price data using different scaling methods.

    Attributes:
        train_size (float): The proportion of the dataset to include in the training split.
        scalers (dict): A dictionary to store the scaler objects for each feature.
    """

    def __init__(self, train_size=0.8):
        """
        Initializes the StockPricePreprocessor with a specified training data size.

        Parameters:
            train_size (float): The proportion of the data to be used for training. Default is 0.8.
        """
        self.train_size = train_size
        self.scalers = {}

    def split_data(self, prices):
        """
        Splits the data into training and test sets.

        Parameters:
            prices (array-like): The array of prices to be split.

        Returns:
            tuple: A tuple containing the training data and test data.
        """
        self.split_index = int(self.train_size * len(prices))
        return prices[:self.split_index], prices[self.split_index:]

    def pre_process_Zscore(self, prices, feature_name: str):
        """
        Applies Z-score normalization to the given prices.

        Parameters:
            prices (array-like): The array of prices to be scaled.
            feature_name (str): The name of the feature for storing the associated scaler.

        Returns:
            tuple: A tuple containing the scaled training data and scaled test data, each as 1D array, respectively.
        """
        prices_np = np.asarray(prices)
        training_data, test_data = self.split_data(prices_np)
        scaler = StandardScaler()
        training_data_scaled = scaler.fit_transform(training_data.reshape(-1, 1))
        test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

        self.scalers[feature_name] = scaler  # Store the scaler
        return training_data_scaled.ravel(), test_data_scaled.ravel()

    def pre_process_maxmin(self, prices, feature_name: str):
        """
        Applies Min-Max normalization to the given prices.

        Parameters:
            prices (array-like): The array of prices to be scaled.
            feature_name (str): The name of the feature for storing the associated scaler.

        Returns:
            tuple: A tuple containing the scaled training data and scaled test data, each as 1D array, respectively.
        """
        prices_np = np.asarray(prices)
        training_data, test_data = self.split_data(prices_np)
        scaler = MinMaxScaler()
        training_data_scaled = scaler.fit_transform(training_data.reshape(-1, 1))
        test_data_scaled = scaler.transform(test_data.reshape(-1, 1))

        self.scalers[feature_name] = scaler  # Store the scaler
        return training_data_scaled.ravel(), test_data_scaled.ravel()

    def inverse_transform(self, scaled_data, feature_name: str):
        """
        Applies the inverse transformation to the scaled data using the scaler associated with the given feature name.

        Parameters:
            scaled_data (array-like): The scaled data to be transformed back to original scale.
            feature_name (str): The name of the feature whose scaler is to be used.

        Returns:
            array-like: The data transformed back to its original scale.

        Raises:
            ValueError: If no scaler is found for the given feature name.
        """
        scaler = self.scalers.get(feature_name)
        if scaler:
            return scaler.inverse_transform(scaled_data.reshape(-1, 1))
        else:
            raise ValueError(f"No scaler found for feature: {feature_name}")
