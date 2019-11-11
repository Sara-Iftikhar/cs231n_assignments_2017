import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, dim = X.shape
  num_class = max(y) + 1
  for i in range(num_train):
    sample = X[i]
    label = y[i]
    y_hat = np.dot(sample, W)
    c = np.max(y_hat)  # for numerical stability
    y_hat -= c
    y_exp = np.exp(y_hat)
    total_exp_y = np.sum(y_exp)
    loss -= np.log(y_exp[label] / total_exp_y)

    # calculate gradients
    for j in range(num_class):
      if j == label:
        dW[:, j] += -sample + (y_exp[label] / total_exp_y) * sample
      else:
        dW[:, j] += (y_exp[j] / total_exp_y) * sample
  loss /= num_train
  loss += reg * np.sum(W ** 2)

  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train, dim = X.shape
  y_hat = np.dot(X, W)
  y_hat -= np.max(y_hat, axis=1, keepdims=True)
  y_exp = np.exp(y_hat)
  loss = np.sum(-np.log(y_exp[range(num_train), y] / np.sum(y_exp, axis=1)))
  loss /= num_train
  loss += reg * np.sum(W ** 2)

  # Compute gradient
  dy = y_exp / np.sum(y_exp, axis=1, keepdims=True)
  dy[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, dy)
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

