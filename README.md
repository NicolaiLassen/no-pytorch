## Table of Contents

## Install

```bash
$ pip install fno-pytorch
```

## Fourier Neural Operator - Pytorch

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

*We may regard the present state of the universe as the effect of its past and the cause of its future. An intellect which at a certain moment would know all forces that set nature in motion, and all positions of all items of which nature is composed, if this intellect were also vast enough to submit these data to analysis, it would embrace in a single formula the movements of the greatest bodies of the universe and those of the tiniest atom; for such an intellect nothing would be uncertain and the future just like the past would be present before its eyes* — Marquis Pierre Simon de Laplace