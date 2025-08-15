# An Open-Source Implementation of the Critical Line Algorithm for Portfolio Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Description

This repository provides a Python implementation of the Critical Line Algorithm (CLA) for portfolio optimization, as described in the paper:

[David H. Bailey, Marcos López de Prado. *An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization*.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2197616)

While the original paper includes code, this implementation addresses the following:

1.  **Compatibility:** The original code may not function correctly with recent Python versions due to its age.
2.  **Pythonic Style:** This implementation is designed to be more familiar and easier to use for Python developers, as the original paper prioritizes language-agnostic code for easy porting.

This implementation aims to provide a readily usable and up-to-date CLA implementation in Python.

## Installation

You can install the package using pip:

```python
from cla_implement import CLA
```

## Quick Start Demo (main.ipynb)
```python

import matplotlib.pyplot as plt
import numpy as np
from cla_implement import CLA

def plot2D(x, y, xLabel='', yLabel='', title='', pathChart=None):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    plt.xticks(rotation=90)
    fig.tight_layout()
    if pathChart is None:
        plt.show()
    else:
        fig.savefig(pathChart, bbox_inches='tight', dpi=200)
    plt.close(fig)

# 1) Path
path = './CLA_Data.csv'

# 2) Load data
data = np.genfromtxt(path, delimiter=',', skip_header=1)  # as numpy array
mean  = data[0]      # shape (n,)
lB    = data[1]      # shape (n,)
uB    = data[2]      # shape (n,)
covar = data[3:]     # shape (n, n)

# 3) Invoke object
cla = CLA(mean, covar, lB, uB)

# 4) Plot frontier
mu, sigma, weights = cla.get_efficient_frontiers(max(100, len(cla.weights) * 10))
plot2D(sigma, mu, 'Risk', 'Expected Excess Return', 'CLA-derived Efficient Frontier')

# 5) Get Maximum Sharpe ratio portfolio
w_sr, sr = cla.get_max_sharpe_port()
print("Max Sharpe:", sr)

# 6) Get Minimum Variance portfolio
w_mv, mv = cla.get_min_var_port()
print("Min Variance:", mv)


```
## Contributing

Contributions are welcome! Here's how you can contribute:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes.
4.  Write tests to ensure your changes work as expected.
5.  Submit a pull request.

> **Note:**  Please follow the project's coding style and conventions.  Include detailed explanations of your changes in the pull request description.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

*   David H. Bailey and Marcos López de Prado for their work on the Critical Line Algorithm.

## Contact

> Juho Kim - [juho13729@gmail.com](juho13729@gmail.com)

## Environment & Dependency
python 3.11.13, numpy 2.3.2, pandas 2.3.1, scipy 1.16.1, matplotlib 3.10.5

