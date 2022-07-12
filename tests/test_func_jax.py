import jax
import jax.numpy as jnp

from adacat.jax import Adacat

def test_cdf_icdf_consistency(key, k, batch_shape, sample_shape):
    params = jax.random.normal(key, (*batch_shape, k * 2))
    d = Adacat(params[..., :k], params[..., k:])

    x = jax.random.uniform(key, sample_shape + batch_shape)
    x_rec = d.icdf(d.cdf(x))

    assert jnp.allclose(x, x_rec, atol=1e-03), jnp.max(jnp.abs((x - x_rec)))

def test_mean(key, k, batch_shape):
    params = jax.random.normal(key, (*batch_shape, k * 2))
    d = Adacat(params[..., :k], params[..., k:])
    x = d.sample(seed=key, sample_shape=(10000000,))

    emp_mean = x.reshape((-1, *batch_shape)).mean(axis=0)

    print("empirical mean:", float(emp_mean), "analytic mean:", float(d.mean()), "diff:", float(jnp.abs(emp_mean - d.mean())))
    
    assert jnp.allclose(emp_mean, d.mean(), atol=1e-02)

def test_entropy(key, k, batch_shape):
    params = jax.random.normal(key, (*batch_shape, k * 2))
    d = Adacat(params[..., :k], params[..., k:])
    x = d.sample(seed=key, sample_shape=(10000000,))

    emp_ent = -d.log_prob(x).reshape((-1, *batch_shape)).mean(axis=0)

    print("empirical entropy:", float(emp_ent), "analytic entropy:", float(d.entropy()), "diff:", float(jnp.abs(emp_ent - d.entropy())))

    assert jnp.allclose(emp_ent, d.entropy(), atol=1e-02)

key = jax.random.PRNGKey(135)

test = test_cdf_icdf_consistency

for test in [test_cdf_icdf_consistency]:
    test(key, 10, (), ())
    test(key, 10, (5,), ())
    test(key, 10, (), (8,))
    test(key, 10, (5,), (8,))
    test(key, 10, (5, 2), (8,))
    test(key, 10, (5, 2), (8, 5, 3, 2))

test_mean(key, 10, ())
test_entropy(key, 10, ())
