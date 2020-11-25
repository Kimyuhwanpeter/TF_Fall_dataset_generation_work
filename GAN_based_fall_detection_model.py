# -*- coding:utf-8 -*-
import tensorflow as tf
# 오토인코더 형식으로하되--> CycleGAN과 같은 style transform이 아닌 오리지널 GAN에서 사용하는 방식으로하자! --> 그러면 당연히 입력은 노이즈가 되어야 하겠지?
def AUTOENCODER(input_shape=(64, 64, 3), weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)   # 64 x 64 x 64

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)   # 32 x 32 x 128

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)   # 16 x 16 x 256

    h = tf.keras.layers.Conv2D(filters=512,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)   # 8 x 8 x 512

    h = tf.keras.layers.Conv2D(filters=2048,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)   # 4 x 4 x 1024
    h_ = h

    h = tf.keras.layers.GlobalMaxPool2D()(h)
    h = tf.keras.layers.Reshape((4, 4, 128))(h) # 4 x 4 x 64
    h = tf.concat([h, h_], 3)   # 4 x 4 x 1088

    h = tf.keras.layers.Reshape((2, 4, 4, 1088))(h) # 2 x 4 x 4 x 32

    h = tf.keras.layers.Conv3DTranspose(filters=256,
                                        kernel_size=(3,3,3),
                                        strides=(2,4,4),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 4 x 16 x 16 x 256

    h = tf.keras.layers.Conv3DTranspose(filters=128,
                                        kernel_size=(3,3,3),
                                        strides=(2,4,4),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 8 x 64 x 64 x 128

    h = tf.keras.layers.Conv3DTranspose(filters=64,
                                        kernel_size=(3,3,3),
                                        strides=(2,2,2),
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU()(h)  # 16 x 128 x 128 x 64

    h = tf.keras.layers.Conv3D(filters=3,
                               kernel_size=(1,7,7),
                               strides=(1,1,1),
                               padding="valid")(h)
    h = tf.keras.layers.Activation("tanh")(h)   # 16 x 122 x 122 x 3
    #h = tf.keras.layers.experimental.preprocessing.Resizing(112, 112)(h)
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