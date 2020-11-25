# -*- coding: utf-8 -*-
import tensorflow as tf

def generator_3D(input_shape=(16, 224, 224, 3), weight_decay=0.00005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding3D((3,3,3))(h)
    h = tf.keras.layers.Conv3D(filters=64,
                               kernel_size=(7,7,7),
                               strides=(1,1,1),
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h) # d 16 x 224 x 224 x 64

    h = tf.keras.layers.Conv3D(filters=96,
                               kernel_size=(3,3,3),
                               strides=(2,2,2),
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()()
    h = tf.keras.layers.LeakyReLU()(h) # d 8 x 112 x 112 x 96

    h = tf.keras.layers.Conv3D(filters=112,
                               kernel_size=(3,3,3),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()()
    h = tf.keras.layers.LeakyReLU()(h) # 4 x 56 x 56 x 112

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(3,3,3),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
    h = tf.keras.layers.BatchNormalization()()
    h = tf.keras.layers.LeakyReLU()(h) # 2 x 28 x 28 x 128

    h = tf.keras.layers.Conv3D(filters=)




    # 변형된 autoencoder 느낌으로가자! 노트에 적은모양대로 레이어를 구성하고, 채널도 동일
    return tf.keras.Model()
