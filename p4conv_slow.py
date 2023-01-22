import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
import math

"""
List of group operations which can be applied to an image
"""
P4_ops = []
for i in range(4):
    P4_ops.append(lambda img: tf.image.rot90(img, k = i))


class P4Conv2D(Layer):
    """
    Convolutional layer which performs equivariant convolution

    Args: 
    kernel_size: (h,w) size of the filters
    num_channels: number of output feature maps (ignoring the factor of S)
    activation: a string, e.g. 'relu'
    padding: can be a string ('same' or 'valid') or a list of numbers
    """

    def __init__(self, num_channels, kernel_size, padding = 'valid', activation = None):
        super().__init__()
        self.num_channels_out = num_channels
        self.filter_size = kernel_size
        self.activation = activation
        self.padding = padding

    
    def build(self,input_shape):
        num_channels_in = input_shape[-1]
        height_filter, width_filter = self.filter_size

        """
        Here, num_channels_in also includes a factor of S, since the K^l and S dimensions have been merged
        """
        filter_shape = (height_filter, width_filter, num_channels_in, self.num_channels_out)

        psi_init = tf.random_normal_initializer()
        psi_init_val = psi_init(shape=filter_shape,dtype='float32')
        self.psi =  tf.Variable(initial_value = psi_init_val, trainable='true')


    def call(self,input_arr):
        num_channels_in = input_arr.shape[-1]
        height_filter, width_filter = self.filter_size

        """
        Expand filter bank by
        (i) making list of rotated filter banks
        (ii) stacking the rotated filter banks into one tensor
        (iii) flattening the filter bank
        """
        psi_rotated_list = []
        psi_rotated_list.append(P4_ops[0](self.psi))
        psi_rotated_list.append(P4_ops[1](self.psi))
        psi_rotated_list.append(P4_ops[2](self.psi))
        psi_rotated_list.append(P4_ops[3](self.psi))

        psi_expanded = tf.stack(psi_rotated_list, axis = -1)
        psi_expanded = tf.reshape(psi_expanded, (height_filter, width_filter, num_channels_in, len(P4_ops)*self.num_channels_out))

        if isinstance(self.padding, str):
            self.padding = self.padding.upper()

        #reshape for convolution
        result = tf.nn.conv2d(
                    input = input_arr,
                    filters = psi_expanded,
                    strides = 1,
                    padding = self.padding,
                    )


        if self.activation is not None:
            #only implement relu
            return tf.nn.relu(result)


#Pool over rotations (makes output invariant)
class P4MaxPooling2D(Layer):
    def __init__(self):
        super().__init__()

    #takes an input with shape None x h x w x num_channels*S
    def call(self,input_arr):
        batch_size, h, w, num_channels_times_S = input_arr.shape
        input_arr = tf.reshape(input_arr, [-1,h,w,num_channels_times_S // len(P4_ops), len(P4_ops)])

        return tf.math.reduce_max(input_arr, axis = -1)

