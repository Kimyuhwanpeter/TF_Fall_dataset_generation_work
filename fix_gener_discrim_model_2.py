# -*- coding:utf-8 -*-
# https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953766
# https://github.com/NVlabs/stylegan/blob/master/training/networks_stylegan.py
# styleGAN

import tensorflow as tf
import numpy as np

def _blur2d(x, f=[1,2,1], normalize=True, flip=False, stride=1):
    assert len(x.shape) == 4 and all(dim is not None for dim in x.shape[1:])

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[3]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0,0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, 1, stride, stride]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NHWC')
    x = tf.cast(x, orig_dtype)
    return x

def nf(stage, fmap_base=8192, fmap_decay=1.0, fmap_max=256): 
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

def pixel_norm(x, epsilon=1e-8):
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)

class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, name):
        super(PixelNorm, self).__init__()

    def call(self, inputs):
        return pixel_norm(inputs)

class RandomNoise(tf.keras.layers.Layer):
    def __init__(self, layer_idx):
        super(RandomNoise, self).__init__()
        
        res = layer_idx // 2 + 2        
        self.layer_idx = layer_idx
        self.noise_shape = [1, 2**res, 2**res, 1]
    
    def build(self, input_shape):
        self.noise = self.add_variable('noise', shape=self.noise_shape, initializer=tf.initializers.zeros(), trainable=False)
        
    def call(self, inputs):
        return self.noise

class AdaInstanceNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5):
        super(AdaInstanceNorm, self).__init__()
        self.epsilon = epsilon

    def call(self, input, style):
        c_mean, c_var = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        c_std = tf.math.rsqrt(c_var + self.epsilon)
        s_mean, s_var = tf.nn.moments(style, axes=[1,2], keep_dims=True)
        s_std = tf.math.rsqrt(s_var + self.epsilon)
        
        return s_std * (input - c_mean) / c_std + s_mean

class Const(tf.keras.layers.Layer):
    def __init__(self):
        super(Const, self).__init__()

    def build(self, shape_):
        self.const = self.add_variable('const', shape=[1,4,4,256])

    def call(self, inputs):
        return tf.tile(self.const, [tf.shape(inputs)[0], 1, 1, 1])

class ApplyNoise(tf.keras.layers.Layer):
    def __init__(self):
        super(ApplyNoise, self).__init__()

    def build(self, shape):
        input_shape = shape[0]
        self.weights_ = self.add_variable('weights', shape=[input_shape[3]], initializer=tf.initializers.zeros())

    def call(self, inputs):
        x, noise = inputs
        
        return x + noise * tf.reshape(self.weights_, [1, 1, 1, -1])
        #return x + noise * tf.keras.layers.Reshape([1,1,self.weights_.shape[-1]])(self.weights_)

class ApplyBias(tf.keras.layers.Layer):
    def __init__(self, lrmul=1.0):
        super(ApplyBias, self).__init__()
        self.lrmul = lrmul

    def build(self, shape):
        self.bias_ = self.add_variable('bias', shape=[shape[3]])

    def call(self, inputs):
        b = self.bias_ * self.lrmul
        if len(inputs.shape) == 2:
            return inputs + b
        return inputs + tf.reshape(b, [1,1,1,-1])
        #return inputs + tf.keras.layers.Reshape([1,1,self.bias_.shape[-1]])(b)

class StyleModApply(tf.keras.layers.Layer):
    def __init__(self):
        super(StyleModApply, self).__init__()
    
    def call(self, inputs):
        x, style = inputs
        style_ = tf.keras.layers.Reshape([1, 2, x.shape[3]])(style)
        #print(style_.shape)
        #style_ = tf.reshape(style, [-1]+[1]*(len(x.shape) - 2) + [2] + [x.shape[3]])
        #a = x * (style_[:,:,0,:] + 1)
        return x * (style_[:,:,0,:] + 1) + style_[:,:,1,:]

def runtime_coef(kernel_size, gain, fmaps_in, fmaps_out, lrmul=1.0):
    # Equalized learning rate and custom learning rate multiplier.
    shape = [kernel_size[0], kernel_size[1], fmaps_in, fmaps_out]
    fan_in = np.prod(shape[:-1]) # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in) # He init
    init_std = 1.0 / lrmul
    return he_std * lrmul 

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

def mapping_G(input_shape=(256), mapping_fmaps=256):

    h = inputs = tf.keras.Input(input_shape)

    h = PixelNorm(name="G_mapping/PixelNorm")(h)

    for i in range(8):
        size_ = 256 if i == 8 - 1 else mapping_fmaps

        h = tf.keras.layers.Dense(size_, use_bias=True)(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    h = tf.expand_dims(h, 1)
    h = tf.tile(h, [1, 14, 1])

    return tf.keras.Model(inputs=inputs, outputs=h)

def synthesis_G(dlatent_size=256, num_layers=14):

    num_channels = 3
    h = inputs = tf.keras.Input(shape=[num_layers, dlatent_size])
    noise_inputs = []
    for i in range(num_layers):
        noise_inputs.append(RandomNoise(i)(h))

    # Do some at the end of each layer
    def layer_op(x, idx):
        x = ApplyNoise()([x, noise_inputs[idx]])
        x = ApplyBias()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        x = InstanceNormalization()(x)

        style = tf.keras.layers.Dense(x.shape[3]*2)(inputs[:, idx])
        x = StyleModApply()([x, style])

        return x
    
    def block(res, x):
        
        upsampled = tf.keras.layers.Conv2DTranspose(filters=nf(res-1),
                                                    kernel_size=3,
                                                    strides=2,
                                                    padding="same",
                                                    use_bias=False)(x)

        x = layer_op(_blur2d(upsampled), res*2-4)

        x = layer_op(tf.keras.layers.Conv2D(filters=nf(res-1),
                                        kernel_size=3,
                                        strides=1,
                                        padding="same",
                                        use_bias=False)(x), res*2-3)
        return x


    h = layer_op(Const()(h), 0)
    h = layer_op(tf.keras.layers.Conv2D(filters=nf(1),
                                        kernel_size=3,
                                        strides=1,
                                        padding="same",
                                        use_bias=False)(h), 1)

    for res in range(3, 8 + 1):
        h = block(res, h)

    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=1,
                               strides=1,
                               padding="same")(h)
    h = tf.keras.layers.Activation("tanh")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)


def StyleGAN_for_D(dlatent_space=256):

    h = inputs = tf.keras.Input([dlatent_space, dlatent_space, 3])

    def block(res, inputs):

        if res >=3: # 8x8 up
            h = tf.keras.layers.Conv2D(filters=nf(res-1),
                                       kernel_size=3,
                                       strides=1,
                                       padding="same",
                                       use_bias=False)(inputs)
            h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
            h = _blur2d(h)
            
            h = tf.keras.layers.Conv2D(filters=nf(res-2),
                                       kernel_size=3,
                                       strides=1,
                                       padding="same")(h)
            h = tf.keras.layers.AvgPool2D(pool_size=(2,2), strides=2, padding="same")(h)
            h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

        else:   # 4x4
            h = tf.keras.layers.Conv2D(filters=nf(res-1),
                                       kernel_size=3,
                                       strides=1,
                                       padding="same")(inputs)
            h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

            h = tf.keras.layers.Flatten()(h)
            h = tf.keras.layers.Dense(nf(res-2))(h)
            h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
            h = tf.keras.layers.Dense(32)(h)

        return h

    h = tf.keras.layers.Conv2D(filters=nf(8-1),
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False)(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)
    for res in range(8, 2, -1):
        h = block(res, h)
    h = block(2, h)

    return tf.keras.Model(inputs=inputs, outputs=h)