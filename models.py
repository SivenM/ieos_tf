import tensorflow as tf


def sepconv(x, num_channels, stride, padding='same'):
    inputs = x.shape[-1]
    conv_1 = tf.keras.layers.Conv2D(inputs*6, kernel_size=1, activation='relu', padding='same')(x)
    dw_layer = tf.keras.layers.DepthwiseConv2D(
                            (3, 3),
                            padding='same',
                            depth_multiplier=1,
                            strides=stride,
                            activation='relu',
                            use_bias=False)(conv_1)
    bn = tf.keras.layers.BatchNormalization()(dw_layer)
    pointwise = tf.keras.layers.Conv2D(num_channels, kernel_size=1, padding='same')(dw_layer)
    return pointwise

def bottleneck(x, t=6):
    inputs = x.shape[-1]
    conv_1 = tf.keras.layers.Conv2D(inputs*t, kernel_size=1, activation='relu', padding='same')(x)
    dw_layer = tf.keras.layers.DepthwiseConv2D(
                            (3, 3),
                            padding='same',
                            depth_multiplier=1,
                            activation='relu',
                            use_bias=False)(conv_1)
    bn = tf.keras.layers.BatchNormalization()(dw_layer)
    pointwise = tf.keras.layers.Conv2D(inputs, kernel_size=1, padding='same')(dw_layer)
    return tf.keras.layers.Concatenate()([x, pointwise])


def mobile_net2_det(input_shape:tuple):
    inputs = tf.keras.Input(shape=input_shape)
    input_conv = tf.keras.layers.Conv2D(32, kernel_size=1, strides=2, padding='same')(inputs)
    x = bottleneck(input_conv, 1)
    x = sepconv(x, 16, 2, 'same')
    x = bottleneck(x)
    x = sepconv(x, 24, 2, 'same')
    x = bottleneck(x)
    x = sepconv(x, 32, 2, 'same')
    x = bottleneck(x)
    x = sepconv(x, 64, 2, 'same')
    x = bottleneck(x)
    x = sepconv(x, 96, 2, 'same')
    x = bottleneck(x)
    x = sepconv(x, 160, 1, 'same')
    x = bottleneck(x)
    x = sepconv(x, 320, 2, 'same')
    x = bottleneck(x)
    x = tf.keras.layers.Conv2D(1280, kernel_size=1, activation='relu', padding='same')(x)
    y = tf.keras.layers.GlobalAveragePooling2D()(x)
    y = tf.keras.layers.Flatten()(y)
    class_output = tf.keras.layers.Dense(2, activation='softmax')(y)
    loc_output = tf.keras.layers.Dense(4)(y)
    return tf.keras.Model(inputs=inputs, outputs=[loc_output, class_output])