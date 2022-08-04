## AdaCat
pip package for adaptive categorical distribution (AdaCat) introduced in [AdaCat: Adaptive Categorical Discretization for Autoregressive Models](https://openreview.net/forum?id=HMzzPOLs9l5). Colab demos are available in [PyTorch](https://colab.research.google.com/drive/1dVOJpiJLXtAK_O68Tt4995dYU5BU-1S0?usp=sharing) and [JAX distrax](https://colab.research.google.com/drive/14KPbk9MUqG3TyOySuZ-vK_fg9Kkld6Bs?usp=sharing). [Project Website](https://colinqiyangli.github.io/adacat/)

- installation: `pip install adacat`
- see local usage in `demo-(jax/torch).py`/`demo-(jax/torch).ipynb`

![alt text for screen readers](adacat_1d.png "Density Estimation with AdaCat")

## Citation
The bibtex is provided below for citation covenience.
```
@inproceedings{
li2022adacat,
title={AdaCat: Adaptive Categorical Discretization for Autoregressive Models},
author={Qiyang Li and Ajay Jain and Pieter Abbeel},
booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
year={2022},
url={https://openreview.net/forum?id=HMzzPOLs9l5}
}
```

## Acknowledgements
Thank Ilya Kostrikov for great help on the JAX implementation of Adacat.
