import numpy as np
import scipy.io
import tensorflow as tf

class VGG:  
    # Define layers of VD-VGG 19 from 
    # http://www.robots.ox.ac.uk/~vgg/research/very_deep/
    LAYERS = (
      'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
      'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
      'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
      'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
      'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
      'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
      'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
      'relu5_3', 'conv5_4', 'relu5_4')


    def __init__(self, vgg_data_path):
        # Load model parameters
        self.model_data = scipy.io.loadmat(vgg_data_path)
        # Get mean pixel (for preprocessing)
        self.mean_pixel = np.mean(self.model_data['normalization'][0][0][0],
                                  axis=(0,1))
        # Extract model weights from model parameters
        self.weights = self.model_data['layers'][0]

    def preprocess(self, image):
        # Return image minus mean pixel for normalization
        return image - self.mean_pixel

    def unprocess(self, image):
        # Re-add mean pixel for viewing
        return image + self.mean_pixel

    # Create VGG and get all layer activations
    def net(self, input_image):
        net = {}
        current_layer = input_image
        for i, layer in enumerate(self.LAYERS):
            if _is_convolution_layer(layer):
                # Extract weights for layer
                kernels, bias = self.weights[i][0][0][0][0]

                # Kernels stored as [width, height, in_chan, out_chan]
                # Tensorflow needs  [height, width, in_chan, out_chan]
                kernels = np.transpose(kernels, (1,0,2,3)).astype('float64')

                # Vectorize bias
                bias = bias.reshape(-1).astype('float64')

                # Compute convolution
                current_layer = _conv_layer(current_layer, kernels, bias)

            elif _is_relu_layer(layer):
                # Perform ReLU
                current_layer = tf.nn.relu(current_layer)

            elif _is_pooling_layer(layer):
                # Perform maxpool
                current_layer = _pooling_layer(current_layer)

            # Save layer activation
            net[layer] = current_layer

        assert len(net) == len(self.LAYERS)

        return net
    
# Define functions to check layer type
def _is_convolution_layer(layer):
    return layer[:4] == 'conv'

def _is_relu_layer(layer):
    return layer[:4] == 'relu'

def _is_pooling_layer(layer):
    return layer[:4] == 'pool'

# Define function to perform convolution
def _conv_layer(in_data, kernel, bias):
    conv = tf.nn.conv2d(in_data, tf.constant(kernel),
                      strides=(1,1,1,1), padding='SAME')
    return tf.nn.bias_add(conv, bias)

# Define function to perform maxpool
def _pooling_layer(in_data):
#     return tf.nn.max_pool(in_data, ksize=(1,2,2,1),
#                         strides=(1,2,2,1), padding='SAME')
    return tf.layers.average_pooling2d(in_data, pool_size=(2,2),
                        strides=(2,2), padding='SAME')