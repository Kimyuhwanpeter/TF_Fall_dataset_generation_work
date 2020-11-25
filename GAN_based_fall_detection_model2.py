# -*- coding:utf-8 -*-
import tensorflow as tf

def AUTOENCODER_v2(input_shape=(16, 112, 112, 3), weight_decay=0.000005):
    # 모델이 너무 얇다 --> 더 추가해야 할 것 같다.
    def residual_connection(x):
        h = tf.keras.layers.ZeroPadding3D((1,1,1))(x)
        h = tf.keras.layers.Conv3D(filters=256,
                                   kernel_size=(3,3,3),
                                   strides=(1,1,1),
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.LeakyReLU()(h)

        h = tf.keras.layers.ZeroPadding3D((1,1,1))(h)
        h = tf.keras.layers.Conv3D(filters=256,
                                   kernel_size=(3,3,3),
                                   strides=(1,1,1),
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        return tf.keras.layers.LeakyReLU()(h + x)


    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding3D((3,3,3))(h)
    h = tf.keras.layers.Conv3D(filters=64,
                               kernel_size=(7,7,7),
                               strides=(1,1,1),
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 16 x 112 x 112 x 64

    h = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), padding="same")(h) # 8 x 56 x 56 x 64
    h_3 = h

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(3,3,3),
                               strides=(1,1,1),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 8 x 56 x 56 x 128

    h = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), padding="same")(h) # 4 x 28 x 28 x 128
    h_2 = h

    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(3,3,3),
                               strides=(1,1,1),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 4 x 28 x 28 x 256

    h = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), padding="same")(h)    # 2 x 14 x 14 x 256

    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(3,3,3),
                               strides=(1,1,1),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h_1 = h
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 2 x 14 x 14 x 256

    h = tf.keras.layers.MaxPool3D(pool_size=(2,2,2), padding="same")(h) # 1 x 7 x 7 x 256
    ##########################################################################################
    for _ in range(5):
        h = residual_connection(h)  # 1 x 7 x 7 x 256
    #h = tf.keras.layers.Conv3D(filters=256,
    #                           kernel_size=(3,3,3),
    #                           strides=(1,1,1),
    #                           padding="same",
    #                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    #h = tf.keras.layers.BatchNormalization()(h)
    #h = tf.keras.layers.LeakyReLU()(h)  # 1 x 7 x 7 x 256

    h = tf.keras.layers.UpSampling3D()(h)   # 2 x 14 x 14 x 256

    h = tf.keras.layers.Conv3D(filters=256,
                               kernel_size=(3,3,3),
                               strides=(1,1,1),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.math.maximum(h_1, h)

    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 2 x 14 x 14 x 256

    h = tf.keras.layers.UpSampling3D()(h)   # 4 x 28 x 28 x 256

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(3,3,3),
                               strides=(1,1,1),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.math.maximum(h_2, h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 4 x 28 x 28 x 128

    h = tf.keras.layers.UpSampling3D()(h)   # 8 x 56 x 56 x 128

    h = tf.keras.layers.Conv3D(filters=64,
                               kernel_size=(3,3,3),
                               strides=(1,1,1),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.math.maximum(h_3, h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 8 x 56 x 56 x 64

    h = tf.keras.layers.UpSampling3D()(h)   # 16 x 112 x 112 x 64

    h = tf.keras.layers.ZeroPadding3D((3,3,3))(h)
    h = tf.keras.layers.Conv3D(filters=3,
                               kernel_size=(7,7,7),
                               strides=(1,1,1),
                               padding="valid")(h) # 16 x 112 x 112 x 3
    h = tf.keras.layers.Activation("tanh")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def discriminator(input_shape=(16, 122, 122, 3), weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding3D((0, 3, 3))(h)
    h = tf.keras.layers.Conv3D(filters=32,
                               kernel_size=(4,4,4),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 8 x 64 x 64 x 32

    h = tf.keras.layers.Conv3D(filters=64,
                               kernel_size=(4,4,4),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 4 x 32 x 32 x 64

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(4,4,4),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 2 x 16 x 16 x 128

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(4,4,4),
                               strides=(2,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 1 x 8 x 8 x 128

    h = tf.keras.layers.Conv3D(filters=128,
                               kernel_size=(4,4,4),
                               strides=(1,2,2),
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 1 x 4 x 4 x 128

    h = tf.keras.layers.Conv3D(filters=32,
                               kernel_size=(4,4,4),
                               strides=(1,1,1),
                               padding="same")(h)
    h = tf.keras.layers.Reshape((16, 32))(h) # 16 x 32
    
    return tf.keras.Model(inputs=inputs, outputs=h)