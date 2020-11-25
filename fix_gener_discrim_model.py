# -*- coding:utf-8 -*-
import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def generator(input_shape=(16, 64, 64, 3), weight_decay=0.000004):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding3D((3,3,3))(h)
    h = tf.keras.layers.Conv3D(filters=64,
                               kernel_size=(7,7,7),
                               strides=(1,1,1),
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 16 x 64 x 64 x 64

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(3,3,3),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 8 x 32 x 32 x 128

    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(3,3,3),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 4 x 16 x 16 x 256

    h = tf.keras.layers.Conv3D(filters=512,
                               kernel_size=(3,3,3),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 2 x 8 x 8 x 512

    h = tf.keras.layers.Conv3D(filters=512,
                               kernel_size=(3,3,3),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 1 x 4 x 4 x 512

    h = tf.keras.layers.AveragePooling3D(pool_size=(1,4,4), strides=(1,1,1), padding="valid")(h) # 1 x 1 x 1 x 512

    h = tf.keras.layers.ConvLSTM2D(filters=256,
                                   kernel_size=(1,1),
                                   strides=(1,1),
                                   padding="same",
                                   use_bias=False,
                                   return_sequences=True,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 1 x 1 x 1 x 512

    #h = tf.expand_dims(h, 1)
    h = tf.keras.layers.Conv3DTranspose(filters=256,
                                        kernel_size=(3,3,3),
                                        strides=(2,2,2),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 2 x 2 x 2 x 256

    h = tf.keras.layers.Conv3DTranspose(filters=256,
                                        kernel_size=(1,3,3),
                                        strides=(1,2,2),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 2 x 4 x 4 x 256

    h = tf.keras.layers.Conv3DTranspose(filters=128,
                                        kernel_size=(1,3,3),
                                        strides=(1,2,2),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 2 x 8 x 8 x 128

    h = tf.keras.layers.Conv3DTranspose(filters=64,
                                        kernel_size=(3,3,3),
                                        strides=(2,2,2),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 4 x 16 x 16 x 64

    h = tf.keras.layers.Conv3DTranspose(filters=64,
                                        kernel_size=(3,3,3),
                                        strides=(2,2,2),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 8 x 32 x 32 x 64

    h = tf.keras.layers.Conv3DTranspose(filters=64,
                                        kernel_size=(3,3,3),
                                        strides=(2,2,2),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 16 x 64 x 64 x 64

    h = tf.keras.layers.ZeroPadding3D((3,3,3))(h)
    h = tf.keras.layers.Conv3D(filters=3,
                               kernel_size=(7,7,7),
                               strides=(1,1,1),
                               padding="valid")(h)
    h = tf.keras.layers.Activation("tanh")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def discriminator(input_shape=(16, 64, 64, 3), weight_decay=0.000004):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv3D(filters=64,
                               kernel_size=(4,4,4),
                               strides=(1,1,1),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 16 x 64 x 64 x 64

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(4,4,4),
                               strides=(1,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 16 x 32 x 32 x 128

    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(4,4,4),
                               strides=(1,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 16 x 16 x 16 x 256

    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(4,4,4),
                               strides=(1,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 16 x 8 x 8 x 256

    h = tf.keras.layers.Conv3D(filters=32,
                               kernel_size=(1,8,8),
                               strides=(1,1,1),
                               padding="valid")(h)
    h = tf.squeeze(h, [2,3])

    return tf.keras.Model(inputs=inputs, outputs=h)