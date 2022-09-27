import abc

import numpy as np

import network.losses as losses

np.random.seed(1)


class ActivationFunction(abc.ABC):

  @abc.abstractmethod
  def derivative(self, x: np.array) -> np.array:
    """The derivative of the activation function."""

  @abc.abstractmethod
  def forward(self, x: np.array) -> np.array:
    """The forward pass of the activation function."""


class ReLU(ActivationFunction):
  """The Rectified Linear Unit activation function."""

  def __init__(self):
    pass

  def forward(self, x):
    return np.maximum(x, 0.0)

  def derivative(self, x):
    return np.where(x < 0, 0.0, 1.0)


class Softmax(ActivationFunction):
  """The Softmax activation function."""

  def __init__(self):
    pass  # TODO

  def forward(self, x):
    # use x - max(x) for numerical stability
    expX = np.exp(x - np.max(x))
    return expX / np.sum(expX, axis=1, keepdims=True)

  def derivative(self, x):
    raise Exception('Unimplemented.')


class Sigmoid(ActivationFunction):
  """The Sigmoid activation function"""

  def __init__(self):
    pass  # TODO

  def forward(self, x):
    return 1.0 / (1.0 + np.exp(-x))

  def derivative(self, x):
    return self.forward(x) * (1 - self.forward(x))


class LinearLayer(object):
  """A single linear layer with an activation function."""

  def __init__(self, input_dim, output_dim, activation='Sigmoid', init_bias=0.0):
    """Initialize the layer using Xavier initialization.

    Note, Xavier initialization is only valid for tanh/sigmoid activations.

    :param input_dim: Dimension of the inputs to the layer.
    :param output_dim: Dimension of the outputs to the layer.
    :param activation: The type of activation function to use.
    :param init_bias: The initial bias of the layer.
    """
    # Xavier initialization
    self.weights = np.random.uniform(low=-1.0 / np.sqrt(input_dim), high=1.0 / np.sqrt(input_dim),
                                     size=(input_dim, output_dim))
    self.biases = np.ones(output_dim) * init_bias

    # Initialize gradient bookkeeping
    self.A = None  # last output after activation
    self.Z = None  # output before activation
    self.dWeights = None
    self.dBiases = None

    if activation == 'ReLU':
      self.activation = ReLU()
    elif activation == "Sigmoid":
      self.activation = Sigmoid()
    elif activation == "Softmax":
      self.activation = Softmax()
    elif activation is None:
      pass
    else:
      raise Exception("%s activation not valid." % activation)

  def forward(self, x):
    x = np.matmul(x, self.weights) + self.biases
    self.Z = x
    x = self.activation.forward(x)
    self.A = x
    return x

  def backward(self, delta, a):
    da = self.activation.derivative(a)  # the derivative of the activation fn
    return np.matmul(delta, self.weights)[:, 1:] * da


class ANN(object):
  """An artificial neural network made of linear layers."""

  def __init__(self, input_dim, hidden_layers_dims, output_dims, final_activation=None, learning_rate=0.005,
               lambda_regularization=0.005):
    """Initialize the network.

    :param input_dim: Dimension of the input.
    :param hidden_layers_dims: List of the hidden layer dimensions.
    :param output_dims: Dimension of the output.
    :param final_activation: The final activation type.
    :param learning_rate: The learning rate for the system.
    :param lambda_regularization: The weight regularization. TODO: Implement.
    """
    self.learning_rate = learning_rate
    self.lambda_regularization = lambda_regularization
    if len(hidden_layers_dims) == 0:
      # Single layer
      self.layers = [LinearLayer(input_dim, output_dims)]
    else:
      # Multi layer
      self.layers = [LinearLayer(input_dim, hidden_layers_dims[0])]
      for i in range(len(hidden_layers_dims) - 1):
        self.layers.append(LinearLayer(hidden_layers_dims[i], hidden_layers_dims[i + 1]))
      self.layers.append(LinearLayer(hidden_layers_dims[-1], output_dims, activation=final_activation))

  def forward(self, x):
    for layer in self.layers:
      x = layer.forward(x)
    return x

  def backward(self, x, y):
    """Compute the backpropagation or backward pass of the network.

    :param x: Input data. shape=(batch_size, input_dim)
    :param y: Output data. shape=(batch_size, output_dim)
    """
    activations = [self.layers[i].A for i in range(len(self.layers))]
    zs = [self.layers[i].Z for i in range(len(self.layers))]

    # backward pass
    # grad of preactivation in last layer
    delta = self.cost_derivative(activations[-1], y)  # batch x output

    # changing dw, db for the output layer
    self.layers[-1].dBiases = np.sum(delta, axis=0)
    self.layers[-1].dWeights = np.dot(activations[-2].T, delta)
    self.layers[-1].dWeights = np.dot(activations[-2].T, delta)
    delta = np.dot(delta, self.layers[-1].weights.T)

    k = list(range(len(self.layers) - 2, -1, -1))
    for l in k:
      # Update the hidden layers.
      z = zs[l]  # batch x hidden
      da = self.layers[l].activation.derivative(z)  # batch x hidden
      delta = delta * da

      self.layers[l].dBiases = np.sum(delta, axis=0)

      if l == 0:  # if first layer
        self.layers[l].dWeights = np.dot(x.T, delta)
      else:
        self.layers[l].dWeights = np.dot(activations[l - 1].T, delta)
      delta = np.dot(delta, self.layers[l].weights.T)  # batch x hidden

  def optimize(self):
    """Optimizes the network given the current calculated weights and biases.
    """
    for layer in self.layers:
      layer.weights = layer.weights - self.learning_rate * layer.dWeights
      layer.biases = layer.biases - self.learning_rate * layer.dBiases
      layer.dWeights = None
      layer.dBiases = None
      layer.A = None
      layer.Z = None

  def train(self, x, y, epochs=5):
    """Trains the network for a number of epochs given the data

    :param x: Input data. shape=(batch_size, input_dim)
    :param y: Output data. shape=(batch_size, output_dim)
    :param epochs: The number of times to train over the dataset.
    :return:
    """
    training_losses = []
    for loop in range(epochs):
      # TODO: shuffle data.
      y_hat = self.forward(x)
      loss = losses.cross_entropy_loss(y_hat, y)
      training_losses.append(loss)
      self.backward(x, y)
      self.optimize()

      if loop % 100 == 0:
        print("Epoch %d, Loss: %2.4f" % (loop, loss))

    return training_losses

  def cost_derivative(self, output_activations, y):
    """Derivative of the cross-entropy loss/cost.

    :param output_activations: The output of the network.
    :param y: The true labels.
    :return: The cost derivative.
    """
    return output_activations - y
