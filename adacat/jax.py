import distrax
from distrax import Categorical
from functools import reduce
import operator

import jax
import jax.numpy as jnp

import chex

Array = chex.Array

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

class Adacat(distrax.Distribution):

    def __init__(self, w_logits: Array, h_logits: Array, smooth_coeff: float = 0., eps: float = 1e-6):
        self.w_logits, self.h_logits = w_logits, h_logits

        self.k = self.w_logits.shape[-1]  # number of components
        assert self.h_logits.shape[-1] == self.k

        batch_shape = self.h_logits.shape[:-1]

        # bin width (x) and height (y) in the cdf
        self.x_sizes = jax.nn.softmax(self.w_logits, axis=-1)
        self.y_sizes = jax.nn.softmax(self.h_logits, axis=-1)

        # log density within each bin
        self.log_ratio = jax.nn.log_softmax(self.h_logits, axis=-1) - jax.nn.log_softmax(self.w_logits, axis=-1)
        self.ratio = jnp.exp(self.log_ratio)

        # prefix sum of width and height
        self.x_cum = jnp.cumsum(self.x_sizes, axis=-1)
        self.y_cum = jnp.cumsum(self.y_sizes, axis=-1)

        # bin boundaries and centers
        self.x_anchors = jnp.concatenate((jnp.zeros_like(self.x_cum[..., 0:1]), self.x_cum[..., :-1], jnp.ones_like(self.x_cum[..., 0:1]),), axis=-1)
        self.x_mids = (self.x_anchors[..., 1:] + self.x_anchors[..., :-1]) / 2.

        self.prefix_first_moment = jnp.cumsum(self.x_mids * self.y_sizes, axis=-1)

        self.smooth_coeff = smooth_coeff
        self.eps = eps

    def _sample_n(self, key, n):
        batch_shape = self.h_logits.shape[:-1]
        length = len(batch_shape)
        y = jax.random.uniform(key, (n,) + batch_shape)
        return self.icdf(y)

    def smooth_log_prob(self, value):
        value = value[..., None]        

        cdfs  = jax.scipy.stats.norm.cdf(self.x_anchors, value, self.smooth_coeff) 
        diffs = (cdfs[..., 1:] - cdfs[..., :-1]) / (cdfs[..., -1:] - cdfs[..., 0:1] + self.eps)  # compute the truncated Gaussian density

        log_prob = (self.log_ratio * diffs).sum(axis=-1)
        return log_prob

    def log_prob(self, value):
        if self.smooth_coeff > 0.:
            return self.smooth_log_prob(value)
        
        batch_shape = self.x_cum.shape[:-1]
        batch_numel = prod(batch_shape)
        length = len(batch_shape)
        
        if length != 0:
            value = jnp.broadcast_to(value, value.shape[:-length] + batch_shape)

        value_fl = jnp.transpose(value.reshape((-1, batch_numel)), (1, 0))
        x_cum_fl = self.x_cum.reshape((batch_numel, -1))

        indices = jnp.clip(jax.vmap(jnp.searchsorted, in_axes=(0, 0), out_axes=0)(x_cum_fl, value_fl), 0, self.k - 1)

        log_prob = jnp.take_along_axis(self.log_ratio.reshape(-1, self.k), indices, axis=-1) 
        log_prob = jnp.transpose(log_prob, (1, 0)).reshape(value.shape)

        return log_prob

    def cdf(self, value):
        batch_shape = self.x_cum.shape[:-1]
        batch_numel = prod(batch_shape)
        length = len(batch_shape)
        
        if length != 0:
            value = jnp.broadcast_to(value, value.shape[:-length] + batch_shape)

        value_fl = jnp.transpose(value.reshape((-1, batch_numel)), (1, 0))
        x_cum_fl = self.x_cum.reshape((batch_numel, -1))

        indices = jnp.clip(jax.vmap(jnp.searchsorted, in_axes=(0, 0), out_axes=0)(x_cum_fl, value_fl), 0, self.k - 1)

        x_cum = jnp.transpose(jnp.take_along_axis(self.x_cum.reshape(-1, self.k), indices, axis=-1), (1, 0)).reshape(value.shape)
        y_cum = jnp.transpose(jnp.take_along_axis(self.y_cum.reshape(-1, self.k), indices, axis=-1), (1, 0)).reshape(value.shape) 
        x_sizes = jnp.transpose(jnp.take_along_axis(self.x_sizes.reshape(-1, self.k), indices, axis=-1), (1, 0)).reshape(value.shape) 
        y_sizes = jnp.transpose(jnp.take_along_axis(self.y_sizes.reshape(-1, self.k), indices, axis=-1), (1, 0)).reshape(value.shape) 
        
        return (y_cum - (x_cum - value) / x_sizes * y_sizes)

    def icdf(self, value):
        batch_shape = self.x_cum.shape[:-1]
        batch_numel = prod(batch_shape)
        length = len(batch_shape)
        
        if length != 0:
            value = jnp.broadcast_to(value, value.shape[:-length] + batch_shape)

        value_fl = jnp.transpose(value.reshape((-1, batch_numel)), (1, 0))
        y_cum_fl = self.y_cum.reshape((batch_numel, -1))

        indices = jnp.clip(jax.vmap(jnp.searchsorted, in_axes=(0, 0), out_axes=0)(y_cum_fl, value_fl), 0, self.k - 1)

        x_cum = jnp.transpose(jnp.take_along_axis(self.x_cum.reshape(-1, self.k), indices, axis=-1), (1, 0)).reshape(value.shape) 
        y_cum = jnp.transpose(jnp.take_along_axis(self.y_cum.reshape(-1, self.k), indices, axis=-1), (1, 0)).reshape(value.shape)
        x_sizes = jnp.transpose(jnp.take_along_axis(self.x_sizes.reshape(-1, self.k), indices, axis=-1), (1, 0)).reshape(value.shape) 
        y_sizes = jnp.transpose(jnp.take_along_axis(self.y_sizes.reshape(-1, self.k), indices, axis=-1), (1, 0)).reshape(value.shape)

        return (x_cum - (y_cum - value) / y_sizes * x_sizes)

    def prob(self, *args, **kwargs):
        return jnp.exp(self.log_prob(*args, **kwargs))

    def mean(self):
        return (self.x_mids * self.y_sizes).sum(axis=-1)

    def median(self):
        return self.icdf(jnp.ones_like(self.x_mids[..., 0]) * 0.5)

    def robust_mean(self, lower_quantile, upper_quantile):
        batch_shape = self.x_cum.shape[:-1]
        batch_numel = prod(batch_shape)
        length = len(batch_shape)
        
        if length != 0:
            lower_quantile = jnp.broadcast_to(lower_quantile, lower_quantile.shape[:-length] + batch_shape)
            upper_quantile = jnp.broadcast_to(upper_quantile, upper_quantile.shape[:-length] + batch_shape)

        lower_value = self.icdf(lower_quantile)
        upper_value = self.icdf(upper_quantile)
        value_shape = lower_quantile.shape

        y_cum_fl = self.y_cum.reshape((batch_numel, -1))
        
        lower_value_fl = jnp.transpose(lower_quantile.reshape((-1, batch_numel)), (1, 0))
        lower_indices = jnp.clip(jax.vmap(jnp.searchsorted, in_axes=(0, 0), out_axes=0)(y_cum_fl, lower_value_fl), 0, self.k - 1)
        upper_value_fl = jnp.transpose(upper_quantile.reshape((-1, batch_numel)), (1, 0))
        upper_indices = jnp.clip(jax.vmap(jnp.searchsorted, in_axes=(0, 0), out_axes=0)(y_cum_fl, upper_value_fl), 0, self.k - 1)

        lower_prefix = jnp.transpose(jnp.take_along_axis(self.prefix_first_moment.reshape(-1, self.k), lower_indices, axis=-1), (1, 0)).reshape(value_shape) 
        upper_prefix = jnp.transpose(jnp.take_along_axis(self.prefix_first_moment.reshape(-1, self.k), upper_indices, axis=-1), (1, 0)).reshape(value_shape) 

        lower_y_cum = jnp.transpose(jnp.take_along_axis(self.y_cum.reshape(-1, self.k), lower_indices, axis=-1), (1, 0)).reshape(value_shape) 
        upper_y_cum = jnp.transpose(jnp.take_along_axis(self.y_cum.reshape(-1, self.k), upper_indices, axis=-1), (1, 0)).reshape(value_shape) 
        lower_x_cum = jnp.transpose(jnp.take_along_axis(self.x_cum.reshape(-1, self.k), lower_indices, axis=-1), (1, 0)).reshape(value_shape) 
        upper_x_cum = jnp.transpose(jnp.take_along_axis(self.x_cum.reshape(-1, self.k), upper_indices, axis=-1), (1, 0)).reshape(value_shape) 

        fm = upper_prefix - lower_prefix \
            + (lower_y_cum - lower_quantile) * (lower_value + lower_x_cum) / 2. \
            - (upper_y_cum - upper_quantile) * (upper_value + upper_x_cum) / 2.

        return fm / (upper_quantile - lower_quantile)
        
    def entropy(self):
        return -(self.log_ratio * self.y_sizes).sum(axis=-1)

    @property
    def event_shape(self):
        return ()
