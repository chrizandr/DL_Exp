import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
      scores = np.dot(X[i], W)
      norm_scores = scores - max(scores)
      loss_i = -1* norm_scores[y[i]] + np.log(np.sum(np.exp(norm_scores)))
      loss += loss_i
      for j in range(W.shape[1]):
          output = np.exp(norm_scores[j])/np.sum(np.exp(norm_scores))
          if y[i] == j:
              dW[:,j] += (-1 + output) * X[i]
          else:
              dW[:,j] += output * X[i]
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)
  dW /= X.shape[0]
  dW += reg * W
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

  scores = np.dot(X, W)
  norm_score = scores - np.max(scores , axis = 1).reshape(-1,1)
  output = np.exp(norm_score)/np.sum(np.exp(norm_score), axis=1).reshape(-1,1)
  loss = np.sum(-1 * np.log(output[np.arange(X.shape[0]), y]))

  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)

  S = output.copy()
  S[np.arange(X.shape[0]), y] += -1
  dW = np.dot((X.T), S)

  dW = dW/X.shape[0] + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
