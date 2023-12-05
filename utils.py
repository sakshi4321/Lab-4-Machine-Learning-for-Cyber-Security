import numpy as np
import tensorflow as tf


class G(tf.keras.Model):
    def __init__(self, B, B_prime):
        super(G, self).__init__()
        self.B = B
        self.B_prime = B_prime

    def predict(self, data):
        """
        Predicts the class label for the input data.
        Args:
            data: Input data in one line
        Returns: 
            res: Prediction result in one line
        - Predicts class label using self.B(data)
        - Predicts class label using self.B_prime(data) 
        - Compares the predictions and sets the prediction vector or flags disagreement
        - Returns the prediction result vector
        """
        y_pred = self.B(data)
        y = np.argmax(y_pred, axis=1)
        y_prime = np.argmax(self.B_prime(data), axis=1)
        res = np.zeros((y.shape[0], 1284))
        for i in range(y.shape[0]):
            if y[i] == y_prime[i]:
                res[i, :-1] = y_pred[i, :]
            else:
                res[i, 1283] = 1
        return res

  
    def call(self, data):
        """
        Predicts the class and calculates the confidence for each sample
        Args:
            data: Input data in one line
        Returns: 
            res: Prediction result in one line
        1. Predicts the class using self.B(data)
        2. Finds the class with highest probability
        3. Predicts again using self.B_prime(data) 
        4. Fills the prediction vector with original probabilities if classes match, else sets the error flag
        """
        y_pred = self.B(data)
        y = np.argmax(y_pred, axis=1)
        y_prime = np.argmax(self.B_prime(data), axis=1)
        res = np.zeros((y.shape[0], 1284))
        for i in range(y.shape[0]):
            if y[i] == y_prime[i]:
                res[i, :-1] = y_pred[i, :]
            else:
                res[i, 1283] = 1
        return res


