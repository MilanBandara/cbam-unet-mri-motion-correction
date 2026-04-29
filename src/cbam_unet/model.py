"""MR^2-AttUNet architecture for MRI motion artifact correction.

Implements the multi-resolution U-Net backbone with a Convolutional Block
Attention Module (CBAM) inserted in each encoder block, as described in
the accompanying paper. The image input is downsampled at three resolutions
and re-injected as additional context into successive encoder stages.

The model is branded MR^2-AttUNet (Motion Removal x Multi-Resolution Attention
U-Net) in code-facing artefacts; the paper refers to the same architecture as
the "CBAM-U-Net" approach.
"""

from __future__ import annotations

import keras.ops as ops
from keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dense,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Input,
    Layer,
    MaxPooling2D,
    Multiply,
    Resizing,
)
from keras.models import Model


class AttentionBlock(Layer):
    """Convolutional Block Attention Module (CBAM).

    Applies channel attention followed by spatial attention to a 4-D feature
    map. ``ratio`` controls the bottleneck size of the channel-attention MLP.
    """

    def __init__(self, ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.channel_dim = channel
        self.dense1 = Dense(channel // self.ratio, activation="relu", use_bias=False)
        self.dense2 = Dense(channel, use_bias=False)
        self.conv = Conv2D(1, kernel_size=7, padding="same", activation="sigmoid")
        self.gap = GlobalAveragePooling2D()
        self.gmp = GlobalMaxPooling2D()
        self.sigmoid = Activation("sigmoid")
        self.multiply_ch = Multiply()
        self.concat_sp = Concatenate(axis=-1)
        self.multiply_sp = Multiply()
        super().build(input_shape)

    def call(self, inputs):
        x = self.channel_attention_module(inputs)
        x = self.spatial_attention_module(x)
        return x

    def channel_attention_module(self, x):
        x1 = self.gap(x)
        x1 = self.dense1(x1)
        x1 = self.dense2(x1)
        x2 = self.gmp(x)
        x2 = self.dense1(x2)
        x2 = self.dense2(x2)
        combined = x1 + x2
        combined = self.sigmoid(combined)
        combined = ops.reshape(combined, (-1, 1, 1, self.channel_dim))
        return self.multiply_ch([x, combined])

    def spatial_attention_module(self, x):
        avg_pool = ops.mean(x, axis=-1, keepdims=True)
        max_pool = ops.max(x, axis=-1, keepdims=True)
        combined = self.concat_sp([avg_pool, max_pool])
        combined = self.conv(combined)
        return self.multiply_sp([x, combined])

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


def Encoder_Block(
    inputs,
    downsampled_features=None,
    filters: int | None = None,
    kernel_size=(3, 3),
    padding: str = "same",
    activation: str = "relu",
    initial_block: bool = False,
):
    """Encoder stage: two conv-BN-act, CBAM, fuse with downsampled context, then pool."""

    x1 = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(inputs)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation)(x1)
    x1 = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation(activation)(x1)

    attention = AttentionBlock()
    x2 = attention(x1)

    if initial_block:
        x = Concatenate()([x1, x2])
    else:
        x = Concatenate()([x1, x2, downsampled_features])

    for _ in range(2):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)

    downsampled_x = MaxPooling2D((2, 2))(x)

    return downsampled_x, x


def Decoder_Block(
    inputs,
    skip_features,
    filters: int,
    kernel_size=(3, 3),
    padding: str = "same",
    activation: str = "relu",
):
    """Decoder stage: transposed conv, concat skip, two conv-BN-act."""

    x = Conv2DTranspose(
        filters=filters, kernel_size=kernel_size, strides=(2, 2), padding=padding
    )(inputs)
    x = Concatenate()([skip_features, x])
    for _ in range(2):
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    return x


def Output_Block(inputs, activation: str = "sigmoid", padding: str = "same"):
    """1x1 conv producing a single-channel image in [0, 1]."""

    return Conv2D(1, (1, 1), padding=padding, activation=activation)(inputs)


def artifact_removal_model(image_resolution: int = 256, kernel_size=(3, 3)) -> Model:
    """Build MR^2-AttUNet, the multi-resolution CBAM U-Net.

    Parameters
    ----------
    image_resolution
        Side length of the (square) input image in pixels.
    kernel_size
        Convolutional kernel size used throughout the network.
    """

    input_1 = Input(shape=(image_resolution, image_resolution, 1))

    x, s1 = Encoder_Block(input_1, filters=128, initial_block=True)

    input_2 = Resizing(image_resolution // 2, image_resolution // 2)(input_1)
    x, s2 = Encoder_Block(input_2, downsampled_features=x, filters=64)

    input_3 = Resizing(image_resolution // 4, image_resolution // 4)(input_1)
    _, x = Encoder_Block(input_3, downsampled_features=x, filters=32)

    x = Decoder_Block(x, skip_features=s2, filters=64)
    x = Decoder_Block(x, skip_features=s1, filters=128)

    output = Output_Block(x)

    return Model(inputs=input_1, outputs=output, name="mr2_attunet")
