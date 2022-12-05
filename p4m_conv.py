import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
from group_index import P4
import numpy as np
import math


#currently only works with channels_last convention (NHWC)
class P4Conv2D(Layer):
    def __init__(self,num_feature_maps,kernel_size,data_format = 'NHWC',activation=None,first_layer = False):
        super().__init__()
        self.num_feature_maps = num_feature_maps
        self.data_format = data_format
        self.filter_size = kernel_size[0]
        self.activation = activation
        self.first_layer = first_layer

    def build(self,input_shape):
        num_channels_in = input_shape[-1]
        num_channels_out = self.num_feature_maps
        u_size = self.filter_size
        v_size = self.filter_size
        #if first layer, input shape is  None x height x width x num_channels_in
        #if first layer, input shape is None x height x width x |S| x num_channels_in, where S is the point group

        """Assume that we are not dealing with first layer
        """
        if self.first_layer:
            #filter shape is u x v x num_channels_in x num_channels_out (S dimension nonexistent in first_layer)
            filter_shape = (u_size, v_size, num_channels_in, num_channels_out)
        else:
            #filter shape is S x u x v x num_channels_in x num_channels_out (S dimension nonexistent in first_layer)
            filter_shape = (P4.S, u_size, v_size, num_channels_in, num_channels_out)

        psi_init = tf.random_normal_initializer()
        psi_init_val = psi_init(shape=filter_shape,dtype='float32')
        self.psi =  tf.Variable(initial_value = psi_init_val, trainable='true')
        #calculate indices for fast filter-bank augmentation later
        self.indices_for_augment = self.gather_indices()
        

#        bias_init = tf.zeros_initializer()
#        bias_init_val = bias_init(shape = (self.num_feature_maps,),dtype='float32')
#        self.bias = tf.Variable(initial_value = bias_init_val, trainable='true')
    
    def call(self,input_arr):
        num_channels_in = input_arr.shape[-1]
        num_channels_out = self.num_feature_maps
        u_size = self.filter_size
        v_size = self.filter_size
    
        if self.first_layer:
            batch_size, h, w, num_channels_in = input_arr.shape
            """
            Augment filter bank:
            """
            #1. Reshape Filter bank to (n*n) x num_channels_in x num_channels_out
            psi_reshaped = tf.reshape(self.psi, 
                            [u_size * v_size, 
                            num_channels_in, 
                            num_channels_out])
            #do I need to turn gather_indices into tf.Tensor? 
            psi_augmented = tf.gather(psi_reshaped, self.indices_for_augment)
            #reshape indices of psi_augmented to S' x u x v x num_channels_in x num_channels_out
            psi_augmented = tf.reshape(psi_augmented,
                                        [P4.S,
                                        u_size,
                                        v_size,
                                        num_channels_in,
                                        num_channels_out])

            #rearrange indices so that psi_augmented has shape u x v x num_channels_in x S' x num_channels_out
            psi_augmented = tf.transpose(psi_augmented,[1,2,3,0,4])
            #reshape for convolution 
            psi_augmented = tf.reshape(psi_augmented, [u_size, v_size, num_channels_in, P4.S * num_channels_out])
            
            #reshape for convolution
            result = tf.nn.conv2d(
                        input = input_arr,
                        filters = psi_augmented,
                        strides = 1,
                        padding = 'VALID',
                        data_format = self.data_format)
        #not first layer
        else:
            #reshape input to None x h x w x S*num_channels_in for later convolution
            batch_size, h, w, s, num_channels_in = input_arr.shape
            input_arr = tf.reshape(input_arr,[-1,h,w,P4.S * num_channels_in])

            """
            Augment filter bank:
            """
            #1. Reshape Filter bank to (S*n*n) x num_channels_in x num_channels_out
            psi_reshaped = tf.reshape(self.psi, 
                            [P4.S *self.filter_size * self.filter_size, 
                            num_channels_in, 
                            num_channels_out])
            #do I need to turn gather_indices into tf.Tensor? 
            psi_augmented = tf.gather(psi_reshaped, self.indices_for_augment)
            #reshape indices of psi_augmented to S' x S x u x v x num_channels_in x num_channels_out
            psi_augmented = tf.reshape(psi_augmented,
                                        [P4.S,
                                        P4.S,
                                        u_size,
                                        v_size,
                                        num_channels_in,
                                        num_channels_out])

            #rearrange indices so that psi_augmented has shape u x v x S x num_channels_in x S' x num_channels_out
            psi_augmented = tf.transpose(psi_augmented,[2,3,1,4,0,5])
            #reshape for convolution
            psi_augmented = tf.reshape(psi_augmented, [u_size, v_size, P4.S* num_channels_in, P4.S * num_channels_out])
            
            #reshape for convolution
            result = tf.nn.conv2d(
                        input = input_arr,
                        filters = psi_augmented,
                        strides = 1,
                        padding = 'VALID',
                        data_format = self.data_format)

        #reshape output of convolution for interpretability
        h_out,w_out = result.shape[1],result.shape[2]
        result = tf.reshape(result,[-1,h_out,w_out, P4.S, num_channels_out])


        if self.activation is not None:
            #only implement relu
            return tf.nn.relu(result)
        ##reshape output of convolution to None x height_new x width_new x num_feature_maps x S
        #feature_map_size = height - self.filter_size + 1
        #return tf.reshape(result, [-1,feature_map_size,feature_map_size,self.num_feature_maps,len(pt_group)])

    def gather_indices(self): #not first layer
        u_size = self.filter_size
        v_size = self.filter_size
        if self.first_layer:
            #1. Create list of indices for tf.gather
            #for s' in range(S)
            #    for num in range(n*n):
            #       i,j = f(num)
            #       obtain u and v
            #       obtain r-bar, u-bar, v-bar
            #       obtain i-bar, j-bar
            #       use r-bar, i-bar, j-bar ---> num'

            #num = i * v_size + j
            idx_list = np.arange(u_size * v_size, dtype = int)
            i_list = idx_list // v_size
            j_list = idx_list % v_size
            u_list = i_list - u_size //2
            v_list = j_list - v_size //2
            array_ruv = np.array([np.zeros(len(idx_list)),u_list, v_list]).T

            gather_indices = []

            for s_prime in range(P4.S):       
                mat_ruv = P4().idx_to_mat(array_ruv)
                mat_s_prime = P4.idx_to_mat([[s_prime,0,0]])
                mat_new = np.matmul(np.linalg.inv(mat_s_prime),mat_ruv)
                array_ruv_bar = P4.mat_to_idx(mat_new)
                #obtain 3 lists from collective list
                r_bar_list = array_ruv_bar[:,0] #should only be zero
                u_bar_list = array_ruv_bar[:,1]
                v_bar_list = array_ruv_bar[:,2]
                i_bar_list = u_bar_list + u_size //2
                j_bar_list = v_bar_list + v_size //2
                idx_bar_list = i_bar_list * v_size + j_bar_list
                gather_indices.append(idx_bar_list)

            gather_indices = np.array(gather_indices)
            return gather_indices

        #not first layer
        else:
            #2. Create list of indices for tf.gather
            #for s' in range(S)
            #    for num in range(S*n*n):
            #       r,i,j = f(num)
            #       obtain u and v
            #       obtain r-bar, u-bar, v-bar
            #       obtain i-bar, j-bar
            #       use r-bar, i-bar, j-bar ---> num'

            #num = r * u_size * v_size + i * v_size + j
            idx_list = np.arange(P4.S * u_size * v_size, dtype = int)
            r_list = idx_list // (u_size * v_size)
            i_list = (idx_list - r_list * (u_size * v_size)) // v_size
            j_list = idx_list % v_size
            u_list = i_list - u_size //2
            v_list = j_list - v_size //2
            array_ruv = np.array([r_list, u_list, v_list]).T

            gather_indices = []

            for s_prime in range(P4.S):       
                mat_ruv = P4.idx_to_mat(array_ruv)
                mat_s_prime = P4.idx_to_mat([[s_prime,0,0]])
                mat_new = np.matmul(np.linalg.inv(mat_s_prime),mat_ruv)
                array_ruv_bar = P4.mat_to_idx(mat_new)
                #obtain 3 lists from collective list
                r_bar_list = array_ruv_bar[:,0]
                u_bar_list = array_ruv_bar[:,1]
                v_bar_list = array_ruv_bar[:,2]
                i_bar_list = u_bar_list + u_size //2
                j_bar_list = v_bar_list + v_size //2
                idx_bar_list = r_bar_list * u_size * v_size + i_bar_list * v_size + j_bar_list
                gather_indices.append(idx_bar_list)

            gather_indices = np.array(gather_indices)
            return gather_indices

#Pooling
class P4MaxPooling2D(Layer):
    def __init__(self,pool_size, strides = None, padding='VALID'):
        super().__init__()
        self.pool_size = pool_size
        if strides is None:
            self.strides = self.pool_size
        else:
            self.strides = strides
        self.padding = padding

    #takes an input with shape None x h x w x S x num_channels 
    def call(self,input_arr):
        batch_size, h, w, S, num_channels = input_arr.shape
        input_arr = tf.reshape(input_arr, [-1,h,w,S*num_channels])
        result = tf.nn.max_pool2d(input_arr,ksize = self.pool_size, strides = self.strides, padding = self.padding)
        h_out,w_out = result.shape[1],result.shape[2]

        return tf.reshape(result, [-1,h_out,w_out,S,num_channels])
#Padding
class P4ZeroPadding2D(Layer):
    def __init__(self,padding=(1,1)):
        super().__init__()
        self.padding = padding

    #takes an input with shape None x h x w x S x num_channels 
    def call(self,input_arr):
        batch_size, h, w, S, num_channels = input_arr.shape
        input_arr = tf.reshape(input_arr, [-1,h,w,S*num_channels])
        result = layers.ZeroPadding2D(self.padding)(input_arr)
        h_out,w_out = result.shape[1],result.shape[2]

        return tf.reshape(result, [-1,h_out,w_out,S,num_channels])
