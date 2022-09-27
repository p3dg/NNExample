# Classifier

The approach is to create a MLP with sigmoid activations. A class was coded up for the network mimicking the 
structure of typical ML frameworks, with a predict, calculate gradient, then optimize step. This uses
cross entropy loss in order to train the network.

The data is preprocessed for one-hot encoding.

Some improvements to be made
* Shuffle data and take minibatches.
* Check support of other activation functions.
* Add weight penalty (lambda regularization parameter is unused).
* Testing.
* Better docstrings.

## Installation and use
```
conda create --name net --file requirements.txt
conda activate net
python run_classifier.py
```