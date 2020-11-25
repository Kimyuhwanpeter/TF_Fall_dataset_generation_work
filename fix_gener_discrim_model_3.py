# -*- coding:utf-8 -*-
import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
 
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

class Group_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, weight_decay, n_group):
        super(Group_Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = self._strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.n_group = n_group

    def build(self, input_shape):
        input_shape = input_shape
        input_channel = input_shape[3]
        self._strides = [1, self._strides, self._strides, 1]
        kernel_shape = (self.kernel_size, self.kernel_size) + (input_channel // self.n_group, self.filters)
        
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            dtype=tf.float32)
        if self.use_bias:
            self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32)
        else:
            self.bias = None

        self.groupConv = lambda i, k: tf.nn.conv2d(i,
                                                   k,
                                                   strides=self._strides,
                                                   padding=self.padding,
                                                   dilations=(1,1))

    def call(self, inputs):

        if self.n_group == 1:
            outputs = self.groupConv(inputs, self.kernel)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=self.n_group, value=inputs)
            weightGroups = tf.split(axis=3, num_or_size_splits=self.n_group, value=self.kernel)
            convGroups = [self.groupConv(i, k) for i, k in zip(inputGroups, weightGroups)]
            outputs = tf.concat(convGroups, 3)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs

def block(previous_input,current_input, filters, weight_decay, i):

    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(current_input)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    
    if i % 2 == 0:
        h = Group_Conv2D(filters=filters,
                        kernel_size=3,
                        strides=1,
                        padding="SAME",
                        use_bias=False,
                        weight_decay=weight_decay,
                        n_group=32)(h)
        #h = tf.keras.layers.Conv2D(filters=filters,
        #                           kernel_size=3,
        #                           strides=1,
        #                           padding="same",
        #                           use_bias=False,
        #                           kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h + previous_input)
    if i % 2 == 1:
        h = tf.keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=1,
                                            padding="same",
                                            use_bias=False,
                                            depthwise_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h + previous_input)

    h = tf.keras.layers.Conv2D(filters=filters*2,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    if i == 0:
        h = tf.keras.layers.ReLU()(h)
    else:
        h = tf.keras.layers.ReLU()(h + current_input)

    return h

def fix_generator(input_shape=(256, 256, 3), weight_decay=0.00005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h) # 256 x 256 x 64

    h = tf.keras.layers.Conv2D(filters=128,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h) # 128 x 128 x 128

    h = tf.keras.layers.Conv2D(filters=256,
                               kernel_size=3,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    previous_input = h
    h = tf.keras.layers.ReLU()(h) # 64 x 64 x 256

    for i in range(8):
        h = block(previous_input, h, 256, weight_decay, i)  # 64 x 64 x 512

    h = tf.keras.layers.Conv2DTranspose(filters=256,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h) # 128 x 128 x 256

    h = tf.keras.layers.Conv2DTranspose(filters=128,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h) # 256 x 256 x 128

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h) # 256 x 256 x 64

    h = tf.concat([inputs, h], 3)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7,
                               strides=1,
                               padding="valid")(h)
    h = tf.keras.layers.Activation("tanh")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def fix_discriminator(input_shape=(256, 256, 3), weight_decay=0.00005):

    h = inputs = tf.keras.Input(input_shape)
    dim = 64
    dim_ = dim
    # 1
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(3 - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)


    return tf.keras.Model(inputs=inputs, outputs=h)