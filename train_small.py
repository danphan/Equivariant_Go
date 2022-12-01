from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.oneplane import OnePlaneEncoder

from dlgo.networks import small
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.experimental.numpy import moveaxis

from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, ZeroPadding2D

from p4m_conv import P4Conv

"""
Change Channels First to channels Last
"""
class Flip(Layer):
    def __init__(self):
        super(Flip,self).__init__(name='flip')
    def call(self,input_tensor):
        return moveaxis(input_tensor, [1,2,3],[3,1,2]) 

go_board_rows, go_board_cols = 19, 19
num_classes = go_board_rows * go_board_cols
num_games = 300

encoder = OnePlaneEncoder((go_board_rows, go_board_cols))

#to loop through zip files and create features/labels for training
processor = GoDataProcessor(encoder=encoder.name())

#create generator to loop through training/testing data
generator = processor.load_go_data('train', num_games, use_generator=True)
test_generator = processor.load_go_data('test',num_games, use_generator= True)
print(generator.get_num_samples(),'moves in one epoch')
print(generator.get_num_samples()/128,'steps per epoch')

#define keras model
input_shape = (encoder.num_planes, go_board_rows, go_board_cols)
network_layers = small.layers(input_shape)
model = Sequential()
model.add(Flip())
model.add(ZeroPadding2D(padding=3, input_shape=input_shape, data_format='channels_last'))
model.add(P4Conv(num_feature_maps = 48, filter_size = 7))
model.add(Activation('relu'))
for layer in network_layers:
    model.add(layer)
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=CategoricalCrossentropy(), optimizer='adam',metrics=['accuracy'])

model.build((None,*input_shape))

#print model summary
print(model.summary())
#
##train model
#epochs = 50
#batch_size = 128
#model.fit(
#    generator.generate(batch_size, num_classes),
#    steps_per_epoch = generator.get_num_samples()//batch_size,
#    epochs = epochs,
#    validation_data = test_generator.generate(batch_size, num_classes),
#    validation_steps = test_generator.get_num_samples()//batch_size,
#    callbacks=[
#    ModelCheckpoint('checkpoints/small_model_epoch_{epoch}.h5')
#    ]) #callback stores model at each epoch
##model.fit(
##    generator.generate(batch_size, num_classes),
##    epochs = epochs,
##    steps_per_epoch = generator.get_num_samples()/batch_size,
##    validation_data = test_generator.generate(batch_size, num_classes),
##    validation_steps = test_generator.get_num_samples()/batch_size,
##    callbacks=[
##    ModelCheckpoint('../checkpoints/small_model_epoch_{epoch}.h5')
##    ]) #callback stores model at each epoch
##
##evaluate model
#model.evaluate(
#    test_generator.generate(batch_size,num_classes),
#    steps = test_generator.get_num_samples()/batch_size)
