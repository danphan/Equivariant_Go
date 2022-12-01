import tensorflow as tf
from tensorflow.keras.layers import Layer

#define pt_group here, should be a set/list/dictionary of functions to apply to an array
pt_group = []
for i in range(4):
    pt_group.append(lambda img: tf.image.rot90(img, k=i))

#currently only works with channels_last convention (NHWC)
class P4Conv2D(Layer):
    def __init__(self,num_feature_maps,kernel_size,data_format = 'NHWC',activation=None):
        super().__init__()
        self.num_feature_maps = num_feature_maps
        self.data_format = data_format
        self.filter_size = kernel_size[0]
        self.activation = activation

    def build(self,input_shape):
        #if first layer, input shape is  None x height x width x num_channels
        #if first layer, input shape is None x height x width x num_channels x |S|, where S is the point group

        num_channels_times_S = input_shape[-1] #num_channels
        #filter shape is filter_size x filter_size x num_channels x (num_feature_maps*S)  (S dimension nonexistent in first_layer)
        psi_init = tf.random_normal_initializer()
        psi_init_val = psi_init(shape=(self.filter_size, self.filter_size,num_channels_times_S,self.num_feature_maps),dtype='float32')
        self.psi =  tf.Variable(initial_value = psi_init_val, trainable='true')
        

#        bias_init = tf.zeros_initializer()
#        bias_init_val = bias_init(shape = (self.num_feature_maps,),dtype='float32')
#        self.bias = tf.Variable(initial_value = bias_init_val, trainable='true')
    
    def call(self,input_arr):
        batch_size, height, width, num_channels_times_S = input_arr.shape

        """
        initialize and fill the expanded filter bank psi_augmented
        psi_augmented will have shape filter_size x filter_size x num_channels_times_S x num_feature_maps x S
        """
        #psi_augmented = tf.zeros([self.filter_size, self.filter_size, num_channels*len(pt_group), self.num_feature_maps, len(pt_group)], tf.dtypes.float32)
        #for map_idx in range(self.num_feature_maps):
        #    for rot_idx in range(len(pt_group)):
        #        psi_augmented[:,:,:,channel_idx, rot_idx] = pt_group[rot_idx](psi_reshaped[:,:,:,channel_idx])
        list_tensors = []
        for channel_idx in range(self.num_feature_maps):
            list_channel = []
            for rot_idx in range(len(pt_group)):
                #psi_augmented[:,:,:,channel_idx, rot_idx] = pt_group[rot_idx](self.psi[:,:,:,channel_idx])
                list_channel.append(pt_group[rot_idx](self.psi[:,:,:,channel_idx]))
            list_tensors.append(list_channel)
        list_tensors = [tf.stack(list_channel,axis = -1) for list_channel in list_tensors]
        psi_augmented = tf.stack(list_tensors, axis = -2)
        #reshape psi_augmented so the last two dimensions are combined
        psi_augmented = tf.reshape(psi_augmented,[self.filter_size, self.filter_size, num_channels_times_S, self.num_feature_maps*len(pt_group)])
        result = tf.nn.conv2d(
                    input = input_arr,
                    filters = psi_augmented,
                    strides = 1,
                    padding = 'VALID',
                    data_format = self.data_format)

        if self.activation is not None:
            #only implement relu
            return tf.nn.relu(result)
        ##reshape output of convolution to None x height_new x width_new x num_feature_maps x S
        #feature_map_size = height - self.filter_size + 1
        #return tf.reshape(result, [-1,feature_map_size,feature_map_size,self.num_feature_maps,len(pt_group)])
