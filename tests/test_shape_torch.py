import torch
from adacat.torch import Adacat

def test(k, batch_shape, sample_shape):
    x = Adacat(torch.randn(*batch_shape, k))
    z = x.sample(sample_shape)

    assert z.shape == sample_shape + batch_shape
    assert x.log_prob(z).shape == sample_shape + batch_shape

test(10, (), ())
test(10, (5,), ())
test(10, (), (8,))
test(10, (5,), (8,))
test(10, (5, 2), (8,))
test(10, (5, 2), (8, 5, 3, 2))
