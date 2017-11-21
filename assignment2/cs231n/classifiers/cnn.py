#from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C, H, W = input_dim
        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters*H/2*W/2, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        conv_out, conv_cache = conv_forward_im2col(X, W1, b1, conv_param)
        relu1_out, relu1_cache = relu_forward(conv_out)
        pool_out, pool_cache = max_pool_forward_fast(relu1_out, pool_param)
        affine_relu_out, affine_relu_cache = affine_relu_forward(pool_out, W2, b2)
        affine2_out, affine2_cache = affine_forward(affine_relu_out, W3, b3)
        scores = affine2_out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg*(np.sum(self.params['W1']* self.params['W1'])
                + np.sum(self.params['W2']* self.params['W2'])+np.sum(self.params['W3']* self.params['W3']))

        affine2_dx, affine2_dw, affine2_db = affine_backward(dscores, affine2_cache)
        grads['W3'] = affine2_dw + self.reg * self.params['W3']
        grads['b3'] = affine2_db

        affine1_dx, affine1_dw, affine1_db = affine_relu_backward(affine2_dx, affine_relu_cache)
        grads['W2'] = affine1_dw + self.reg * self.params['W2']
        grads['b2'] = affine1_db

        pool_dx = max_pool_backward_fast(affine1_dx, pool_cache)
        relu_dx = relu_backward(pool_dx, relu1_cache)
        conv_dx, conv_dw, conv_db = conv_backward_im2col(relu_dx, conv_cache)
        grads['W1'] = conv_dw + self.reg * self.params['W1']
        grads['b1'] = conv_db

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class TwoLayerConvNet_TwoLayerAffine(object):
  """
  A Four-layer convolutional network with the following architecture:

  conv - relu - conv - relu - pool - affine - relu - affine - [softmax or SVM]

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, num_filters2 = 64, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # Initialize                                                               #
    ############################################################################
    C, H, W = input_dim
    self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(0, weight_scale, (num_filters2, num_filters, filter_size, filter_size))
    self.params['b2'] = np.zeros(num_filters2)
    self.params['W3'] = np.random.normal(0, weight_scale, (num_filters2*H/2*W/2, hidden_dim))
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['W4'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    self.params['b4'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    filter_size2 = W2.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    conv2_param = {'stride': 1, 'pad': (filter_size2 - 1) / 2}
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    # conv - relu - conv - relu - pool - affine - relu - affine - [softmax or SVM]

    ############################################################################
    # Forward pass                                                             #
    ############################################################################
    conv1_out, conv1_cache = conv_forward_im2col(X, W1, b1, conv_param)
    relu1_out, relu1_cache = relu_forward(conv1_out)
    conv2_out, conv2_cache = conv_forward_im2col(relu1_out, W2, b2, conv2_param)
    relu2_out, relu2_cache = relu_forward(conv2_out)
    pool_out, pool_cache = max_pool_forward_fast(relu2_out, pool_param)
    affine_relu_out, affine_relu_cache = affine_relu_forward(pool_out, W3, b3)
    affine2_relu_out, affine2_relu_cache = affine_relu_forward(affine_relu_out, W4, b4)
    scores = affine2_relu_out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg*(np.sum(self.params['W1']* self.params['W1'])
            + np.sum(self.params['W2']* self.params['W2'])+np.sum(self.params['W3']* self.params['W3'])
            + np.sum(self.params['W4']* self.params['W4']))

    affine2_relu_dx, affine2_relu_dw, affine2_relu_db = affine_relu_backward(dscores, affine2_relu_cache)
    grads['W4'] = affine2_relu_dw + self.reg * self.params['W4']
    grads['b4'] = affine2_relu_db

    affine_relu_dx, affine_relu_dw, affine_relu_db = affine_relu_backward(affine2_relu_dx, affine_relu_cache)
    grads['W3'] = affine_relu_dw + self.reg * self.params['W3']
    grads['b3'] = affine_relu_db

    pool_dx = max_pool_backward_fast(affine_relu_dx, pool_cache)

    relu2_dx = relu_backward(pool_dx, relu2_cache)

    conv2_dx, conv2_dw, conv2_db = conv_backward_im2col(relu2_dx, conv2_cache)
    grads['W2'] = conv2_dw + self.reg * self.params['W2']
    grads['b2'] = conv2_db

    relu1_dx = relu_backward(conv2_dx, relu1_cache)
    conv1_dx, conv1_dw, conv1_db = conv_backward_im2col(relu1_dx, conv1_cache)
    grads['W1'] = conv1_dw + self.reg * self.params['W1']
    grads['b1'] = conv1_db
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
