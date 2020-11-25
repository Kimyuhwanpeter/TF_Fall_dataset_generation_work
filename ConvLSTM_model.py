# -*- coding:utf-8 -*-
import tensorflow as tf
#https://arxiv.org/pdf/1907.06571.pdf 참고!
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

def residual(input, weight_decay=0.00002):

    h = tf.keras.layers.ZeroPadding3D((1,1,1))(input)
    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(3,3,3),
                               strides=(1,1,1),
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.ZeroPadding3D((1,1,1))(h)
    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(3,3,3),
                               strides=(1,1,1),
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)

    return tf.keras.layers.ReLU()(input + h)

def generator(input_shape=(16, 112, 112, 3), weight_decay=0.00002):

    h = inputs = tf.keras.Input(input_shape)
    # 비디오를 GAN에 돌려야하는데 어떤식으로 접근해야하는지가 가장 의문이다.
    # 그냥 순수 3D conv로만 설계하자!
    h = tf.keras.layers.ZeroPadding3D((3,3,3))(h)
    h = tf.keras.layers.Conv3D(filters=64,
                               kernel_size=(7,7,7),
                               strides=(1,1,1),
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 16 x 112 x 112 x 64

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(3,3,3),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 8 x 56 x 56 x 128

    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(3,3,3),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 4 x 28 x 28 x 256

    for _ in range(5):
        h = residual(h, weight_decay)   # 4 x 28 x 28 x 256
    
    h = tf.keras.layers.Conv3DTranspose(filters=128,
                                        kernel_size=(3,3,3),
                                        strides=(2,2,2),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 8 x 56 x 56 x 128

    h = tf.keras.layers.Conv3DTranspose(filters=64,
                                        kernel_size=(3,3,3),
                                        strides=(2,2,2),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 16 x 112 x 112 x 64

    h = tf.keras.layers.ZeroPadding3D((3,3,3))(h)
    h = tf.keras.layers.Conv3D(filters=3,
                               kernel_size=(7,7,7),
                               strides=(1,1,1),
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)

    h = tf.nn.tanh(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def discriminator(input_shape=(16, 112, 112, 3), weight_decay=0.00002):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv3D(filters=32,
                               kernel_size=(4,4,4),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 8 x 56 x 56 x 32

    h = tf.keras.layers.Conv3D(filters=64,
                               kernel_size=(4,4,4),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 4 x 28 x 28 x 64

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(4,4,4),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 2 x 14 x 14 x 128

    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(4,4,4),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 1 x 7 x 7 x 256

    h = tf.keras.layers.Conv3D(filters=1,
                               kernel_size=(1, 4, 4),
                               strides=(1, 1, 1),
                               padding="same")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)