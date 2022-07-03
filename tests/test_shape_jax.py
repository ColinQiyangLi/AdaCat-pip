import jax
from adacat.jax import Adacat

def test(key, k, batch_shape, sample_shape):
    params = jax.random.normal(key, (*batch_shape, k * 2))
    x = Adacat(params[..., :k], params[..., k:])
    z = x.sample(seed=key, sample_shape=sample_shape)

    assert z.shape == sample_shape + batch_shape
    assert x.log_prob(z).shape == sample_shape + batch_shape

key = jax.random.PRNGKey(0)

test(key, 10, (), ())
test(key, 10, (5,), ())
test(key, 10, (), (8,))
test(key, 10, (5,), (8,))
test(key, 10, (5, 2), (8,))
test(key, 10, (5, 2), (8, 5, 3, 2))
