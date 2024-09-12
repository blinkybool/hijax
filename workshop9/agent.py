"""
RL agent network.
"""

import functools

import jax
import jax.numpy as jnp
import einops
from jaxtyping import Array, Float, Bool, PRNGKeyArray

import strux

# # # 
# Linear transform


@strux.struct
class LinearTransformParams:
    weights: Float[Array, "num_inputs num_outputs"]


class LinearTransform:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs


    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, rng: PRNGKeyArray) -> LinearTransformParams:
        bound = jax.lax.rsqrt(jnp.float32(self.num_inputs))
        return LinearTransformParams(
            weights=jax.random.uniform(
                key=rng,
                shape=(self.num_inputs, self.num_outputs),
                minval=-bound,
                maxval=+bound,
            ),
        )


    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        p: LinearTransformParams,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ p.weights


# # # 
# Affine transform


@strux.struct
class AffineTransformParams:
    weights: Float[Array, "num_inputs num_outputs"]
    biases: Float[Array, "num_outputs"]


class AffineTransform:
    def __init__(self, num_inputs: int, num_outputs: int):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs


    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, rng: PRNGKeyArray) -> AffineTransformParams:
        bound = jax.lax.rsqrt(jnp.float32(self.num_inputs))
        return AffineTransformParams(
            weights=jax.random.uniform(
                key=rng,
                shape=(self.num_inputs, self.num_outputs),
                minval=-bound,
                maxval=+bound,
            ),
            biases=jnp.zeros(self.num_outputs),
        )


    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        p: AffineTransformParams,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ p.weights + p.biases


# # # 
# Convolution layer


@strux.struct
class ConvolutionParams:
    kernel: Float[Array, "num_channels_out num_channels_in kernel_size kernel_size"]


class Convolution:
    """
    Basic convolution module.

    Fields:

    * num_input_channels: int
            Number of channels in input image.
    * num_output_channels: int
            Number of channels in output image.
    * kernel_size: int (>= 1)
            Width and height of the convolution kernel.
    * stride_size: int (default 1)
            Number of steps to take between convolutions. Row and column.
    * pad_same: bool (default False)
            Whether to pad the input image with enough zeros so that the
            output height/width of the image is the same ('SAME' mode), vs.
            no padding so that the output height/width is smaller ('VALID'
            mode).
    """
    def __init__(
        self,
        num_input_channels: int,
        num_output_channels: int,
        kernel_size: int,
        stride_size: int = 1,
        pad_same: bool = False,
    ):
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.pad_same = pad_same


    @functools.partial(jax.jit, static_argnames=["self"])
    def init(self, rng: PRNGKeyArray) -> ConvolutionParams:
        num_inputs = self.num_input_channels * self.kernel_size**2
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        return ConvolutionParams(
            kernel=jax.random.uniform(
                key=rng,
                shape=(
                    self.kernel_size,
                    self.kernel_size,
                    self.num_input_channels,
                    self.num_output_channels,
                ),
                minval=-bound,
                maxval=+bound,
            ),
        )


    @functools.partial(jax.jit, static_argnames=["self"])
    def __call__(
        self,
        p: ConvolutionParams,
        x: Float[Array, "h_in w_in num_channels_in"],
    ) -> Float[Array, "h_out w_out num_channels_out"]:
        x_1hwc = einops.rearrange(x, 'h w c -> 1 h w c')
        y_1hwc = jax.lax.conv_general_dilated(
            lhs=x_1hwc,
            rhs=p.kernel,
            window_strides=(self.stride_size, self.stride_size),
            padding="SAME" if self.pad_same else "VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        return y_1hwc[0]


# # # 
# Actor--critic network architecture


@strux.struct
class ActorCriticNetworkParams:
    conv0: ConvolutionParams
    convs: ConvolutionParams # ConvolutionParams[num_conv_layers-1]
    dense0: AffineTransformParams
    denses: AffineTransformParams # AffineTransformParams[num_dense_layers-1]
    heads: AffineTransformParams


class ActorCriticNetwork:
    def __init__(
        self,
        obs_height: int,
        obs_width: int,
        obs_channels: int,
        net_channels: int,
        net_width: int,
        num_conv_layers: int,
        num_dense_layers: int,
        num_actions: int,
    ):
        self.conv0 = Convolution(
            num_input_channels=obs_channels,
            num_output_channels=net_channels,
            kernel_size=3,
            pad_same=True,
        )
        self.conv = Convolution(
            num_input_channels=net_channels,
            num_output_channels=net_channels,
            kernel_size=3,
            pad_same=True,
        )
        self.dense0 = AffineTransform(
            num_inputs=obs_height * obs_width * net_channels,
            num_outputs=net_width,
        )
        self.dense = AffineTransform(
            num_inputs=net_width,
            num_outputs=net_width,
        )
        self.heads = AffineTransform(
            num_inputs=net_width,
            num_outputs=num_actions+1, # +1 for value
        )
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers


    def init(self, rng: PRNGKeyArray) -> ActorCriticNetworkParams:
        k1, k2, k3, k4, k5 = jax.random.split(rng, 5)
        return ActorCriticNetworkParams(
            conv0=self.conv0.init(k1),
            convs=jax.vmap(self.conv.init)(
                jax.random.split(k2, self.num_conv_layers-1)
            ),
            dense0=self.dense0.init(k3),
            denses=jax.vmap(self.dense.init)(
                jax.random.split(k4, self.num_dense_layers-1)
            ),
            heads=self.heads.init(k5),
        )


    def __call__(
        self,
        p: ActorCriticNetworkParams,
        x: Float[Array, "obs_height obs_width obs_channels"],
    ) -> tuple[
        Float[Array, "num_actions"],
        Float[Array, ""],
    ]:
        # embed observation with residual CNN
        x = self.conv0(p.conv0, x)
        x, _ = jax.lax.scan(
            lambda x, p: (x + self.conv(p, x), None),
            x,
            p.convs,
        )
        # further with residual dense network
        x = jnp.ravel(x)
        x = self.dense0(p.dense0, x)
        x, _ = jax.lax.scan(
            lambda x, p: (x + self.dense(p, x), None),
            x,
            p.denses,
        )
        
        # apply action/value heads
        y = self.heads(p.heads, x)
        # extract action probs
        action_logits = y[:-1]
        action_probs = jax.nn.softmax(action_logits)
        # extract critic value
        value = y[-1]

        return action_probs, value


