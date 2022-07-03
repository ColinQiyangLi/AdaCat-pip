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

        self.smooth_coeff = smooth_coeff
        self.eps = eps

    def _sample_n(self, key, n):
        batch_shape = self.h_logits.shape[:-1]
        length = len(batch_shape)
        cat_dist = Categorical(logits=self.h_logits)

        key_cat, key_unif = jax.random.split(key, 2)
        indices = cat_dist.sample(seed=key_cat, sample_shape=(n,))

        # shape [ x x x y ] => [y x x x ]
        x_cum_t = self.x_cum.transpose(length, *range(length))
        x_sizes_t = self.x_sizes.transpose(length, *range(length))

        x_right = jnp.take_along_axis(x_cum_t, indices, axis=0)
        x_size = jnp.take_along_axis(x_sizes_t, indices, axis=0)

        x = x_right - x_size * jax.random.uniform(key_unif, shape=x_size.shape)
        return jnp.clip(x, 0., 1.)  # make sure everything is fine

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

    def prob(self, *args, **kwargs):
        return jnp.exp(self.log_prob(*args, **kwargs))

    def mean(self):
        return (self.x_mids * self.y_sizes).sum(axis=-1)

    def entropy(self):
        return -(self.log_ratio * self.y_sizes).sum(axis=-1)

    @property
    def event_shape(self):
        return ()
