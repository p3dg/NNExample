import numpy as np


def cross_entropy_loss(yhat, y):
  """Cross Entropy loss for measuring performance of multi-class classification.

  Assumes one-hot encoding.

  :param yhat: Predicted label
  :param y: Actual label
  :return: Returns the loss
  """
  eps = 1e-8  # numerical stability
  return np.mean(np.sum(-y * np.log(yhat + eps), axis=1))
