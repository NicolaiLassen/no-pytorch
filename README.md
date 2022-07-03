## Table of Contents

## Install

```bash
$ pip install fno-pytorch
```

## Usage

```python
import torch
from fno_pytorch import SpectralConv1d

spec_1d = SpectralConv1d(
            in_channels = 10,
            out_channels = 10,
            modes = 12
            )

x = torch.rand(1, 10, 64)

out = spec_1d(img) # (1, 10, 64)
```
## Resources

## Citations

```bibtex
@misc{li2020fourier,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```