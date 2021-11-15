import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class Model(tf.keras.Model):
  def __init__(self, activation='softmax'):
    super(MyModel, self).__init__()
    self.conv2d1 = Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 3), activation='relu', padding='same')
    self.maxpool2d1 = MaxPool2D(pool_size=(2, 2), data_format='channels_last')
    self.flatten1 = Flatten()
    self.dense1 = Dense(256, activation='relu')
    self.dense2 = Dense(10, activation=activation)

  def call(self, inputs):
    out = self.conv2d1(inputs)
    out = self.maxpool2d1(out)
    out = self.flatten1(out)
    out = self.dense1(out)
    out = self.dense2(out)
    return out
