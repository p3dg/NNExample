import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder

import network.losses as losses
import network.neural_network as nn
from sklearn.metrics import multilabel_confusion_matrix


def load_iris_data(filepath: str, test_fraction: float = 0.2) -> np.array:
  """Loads Iris data from file and formats it for training.

  :param filepath: Path to data file.
  :param test_fraction: Fraction of data used in the held-out test set.
  :return: A dictionary of the test and train data.
  """
  data = pd.read_csv(filepath, names=['sl', 'sw', 'pl', 'pw', 'class'])
  input_data = data.iloc[:, 0:4]
  output_data = data.iloc[:, 4]

  # Convert output labels to numbers
  iris_classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
  for i, iris_class in enumerate(iris_classes):
    output_data = output_data.replace(iris_class, i)

  # Convert integer data to one-hot
  output_data = np.array(output_data)
  onehot_encoder = OneHotEncoder(sparse=False)
  output_data = output_data.reshape(len(output_data), 1)
  output_data = onehot_encoder.fit_transform(output_data)

  # Split the data into testing and training data
  n_test = round(len(data) * test_fraction)
  indices_test = np.random.choice(len(data), n_test, replace=False)

  input_train_data = np.array(copy.deepcopy(input_data))
  input_test_data = input_train_data[indices_test]
  input_train_data = np.delete(input_train_data, indices_test, axis=0)

  output_train_data = np.array(copy.deepcopy(output_data))
  output_test_data = output_train_data[indices_test]
  output_train_data = np.delete(output_train_data, indices_test, axis=0)

  data_dict = {
    "input_train_data": input_train_data,
    "input_test_data": input_test_data,
    "output_train_data": output_train_data,
    "output_test_data": output_test_data,
  }
  return data_dict

def plot_confusion(cf_matrix):
  """Plots a confusion matrix with annotations.

  :param cf_matrix: The confusion matrix array (2x2)
  """
  group_names = ['True Neg','False Pos','False Neg','True Pos']
  group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
  group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
  labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  sn.heatmap(cf_matrix, annot=labels, fmt='')


def main():
  # Load up Iris database
  data = load_iris_data("Iris_data.txt")
  model = nn.ANN(input_dim=4, hidden_layers_dims=[10], output_dims=3, final_activation='Softmax')
  yhat_original = model.forward(data['input_train_data'])
  y = data['output_train_data']

  loss = model.train(data['input_train_data'], data['output_train_data'], epochs=5000)

  # Plotting.
  yhat = model.forward(data['input_test_data'])
  y = data['output_test_data']
  confusion = multilabel_confusion_matrix(y, np.round(yhat))
  plt.figure()
  plt.title('Confusion setosa - Test data')
  plot_confusion(confusion[0])
  plt.figure()
  plt.title('Confusion versicolor - Test data')
  plot_confusion(confusion[1])
  plt.figure()
  plt.title('Confusion virginica - Test data')
  plot_confusion(confusion[2])
  plt.show()

  plt.figure()
  plt.plot(loss)
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.show()


if __name__ == '__main__':
  main()
