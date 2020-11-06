from builtins import range
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = X.dot(W)
    for i in range(num_train):
        z_values = scores[i]
        z_values -= np.max(z_values)
        soft_f = np.exp(z_values)/(np.sum(np.exp(z_values)))
        loss += -np.log(soft_f[y[i]])
        y_hot = np.zeros([num_classes])
        y_hot[y[i]] = 1
        for j in range(num_classes):
            dW[:, j] += (soft_f[j] - y_hot[j])*(X[i])  #Ref https://www.ics.uci.edu/~pjsadows/notes.pdf
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    z_values = X.dot(W)
    z_values -= np.expand_dims(np.max(z_values, axis=1), axis=1)
    soft_f = np.exp(z_values) / np.expand_dims(np.sum(np.exp(z_values),axis = 1), axis = 1)
    values_for_correct_class = soft_f[np.arange(num_train), y]
    loss += -np.sum(np.log(values_for_correct_class))
    loss /= num_train
    loss += reg * np.sum(W * W)
    y_hot = np.zeros([num_train, num_classes])
    y_hot[np.arange(num_train), y] = 1
    dW = (X.T).dot(soft_f - y_hot)
    dW /= num_train
    dW += reg * 2 * W

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
