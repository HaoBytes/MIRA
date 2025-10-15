<div align="center">
  <h2><b>(NeurIPS '25) MIRA: Medical Time Series Foundation Model for Real-World Health Data </b></h2>
</div>

<div align="center">

**[<a href="https://arxiv.org/abs/2506.07584">Paper Page</a>]**

</div>

## Overview

MIRA unifies representation learning across multiple medical time-series datasets and supports zero-shot forecasting for downstream clinical prediction tasks.

**Key features**
- CT-RoPE for continuous-time positional encoding
- Frequency-specialized MoE to adapt across rhythms
- Neural-ODE extrapolation for forecasting at arbitrary timestamps

<p align="center">
  <img src="images/Model_Architecture.png" width="720"/>
</p>

---

## Installation

Install Python 3.10+, and then install the dependencies:

```shell
pip install -r requirements.txt
```

**Note: MIRA requires `transformers==4.40.1` .**
