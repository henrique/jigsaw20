""" Generalized mean pooling layers """
import tensorflow as tf

class GeneralizedMeanPooling1D(tf.keras.layers.Layer):
    def __init__(self, p=3, epsilon=1e-6, name='', **kwargs):
        super().__init__(name, **kwargs)
        self.init_p = p
        self.epsilon = epsilon

    def build(self, input_shape):
        if isinstance(input_shape, list) or len(input_shape) != 3:
            raise ValueError(f'`GeM` pooling layer only allow 1 input with 3 dimensions(b, s, c): {input_shape}')
        self.build_shape = input_shape
        self.p = self.add_weight(
              name='p',
              shape=[1,],
              initializer=tf.keras.initializers.Constant(value=self.init_p),
              regularizer=None,
              trainable=True,
              dtype=tf.float32
              )
        self.built=True

    def call(self, inputs):
        input_shape = inputs.get_shape()
        if isinstance(inputs, list) or len(input_shape) != 3:
            raise ValueError(f'`GeM` pooling layer only allow 1 input with 3 dimensions(b, s, c): {input_shape}')
        return (tf.reduce_mean(tf.abs(inputs**self.p), axis=1, keepdims=False) + self.epsilon)**(1.0/self.p)


class GeneralizedMeanPooling2D(tf.keras.layers.Layer):
    def __init__(self, p=3, epsilon=1e-6, shape=1, **kwargs):
        super().__init__(name, **kwargs)
        self.init_p = p
        self.epsilon = epsilon
        self.shape = shape

    def build(self, input_shape):
        if isinstance(input_shape, list) or len(input_shape) != 4:
            raise ValueError('`GeM` pooling layer only allow 1 input with 4 dimensions(b, h, w, c)')
        self.build_shape = input_shape
        self.p = self.add_weight(
              name='p',
              shape=[self.shape,],
              initializer=tf.keras.initializers.Constant(value=self.init_p),
              regularizer=None,
              trainable=True,
              dtype=tf.float32
              )
        self.built=True

    def call(self, inputs):
        input_shape = inputs.get_shape()
        if isinstance(inputs, list) or len(input_shape) != 4:
            raise ValueError('`GeM` pooling layer only allow 1 input with 4 dimensions(b, h, w, c)')
        return (tf.reduce_mean(tf.abs(inputs**self.p), axis=[1,2], keepdims=False) + self.epsilon)**(1.0/self.p)
