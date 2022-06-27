import torch
from adacat import Adacat

def test_cdf_icdf_consistency(k, batch_shape, sample_shape):
    d = Adacat(torch.randn(*batch_shape, k))

    x = torch.rand(sample_shape + batch_shape)
    x_rec = d.icdf(d.cdf(x))

    assert torch.allclose(x, x_rec, atol=1e-03), (x - x_rec).abs().max()

def test_mean(k, batch_shape):
    d = Adacat(torch.randn(*batch_shape, k))
    x = d.sample((10000000,))

    emp_mean = x.view(-1, *batch_shape).mean(dim=0)

    print("empirical mean:", float(emp_mean), "analytic mean:", float(d.mean), "diff:", float((emp_mean - d.mean).abs()))

    assert torch.allclose(emp_mean, d.mean, atol=1e-02)

def test_entropy(k, batch_shape):
    d = Adacat(torch.randn(*batch_shape, k))
    x = d.sample((10000000,))

    emp_ent = -d.log_prob(x).view(-1, *batch_shape).mean(dim=0)

    print("empirical entropy:", float(emp_ent), "analytic entropy:", float(d.entropy), "diff:", float((emp_ent - d.entropy).abs()))

    assert torch.allclose(emp_ent, d.entropy, atol=1e-02)

def test_cdf(k, batch_shape):
    d = Adacat(torch.randn(*batch_shape, k))

    x = d.sample((10000000,))
    
    emp_mean = ((1. - d.cdf(x)) / d.log_prob(x).exp()).mean(dim=0)

    print("empirical mean (via cdf):", float(emp_mean), "analytic mean:", float(d.mean), "diff:", float((emp_mean - d.mean).abs()))

    assert torch.allclose(emp_mean, d.mean, atol=1e-02)

test = test_cdf_icdf_consistency

for test in [test_cdf_icdf_consistency]:
    test(10, (), ())
    test(10, (5,), ())
    test(10, (), (8,))
    test(10, (5,), (8,))
    test(10, (5, 2), (8,))
    test(10, (5, 2), (8, 5, 3, 2))

test_mean(10, ())
test_entropy(10, ())
test_cdf(10, ())
