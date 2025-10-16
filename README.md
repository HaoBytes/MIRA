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

---
## Data Preparation

### Data format example

Each line represents one sample and must contain at least `sequence` and
`time` fields:

``` json
{"sequence": [1.0, 1.2, 0.8, ...], "time": [0.12, 1.52, 2.31, ...], "mask": [1,1,1,...]}
{"sequence": [5.1, 5.0, 5.3, ...], "time": [1699990000, 1699990600, 1699991200, ...], "mask": [1,1,1,...]}
```
---
## Training


## Inference

```python
import torch
from transformers import AutoModelForCausalLM

context_length = 12
normed_seqs = torch.randn(2, context_length)  # tensor shape is [batch_size, context_length]

model = AutoModelForCausalLM.from_pretrained(
    'MIRA',
    device_map="cpu",  # use "cpu" for CPU inference, and "cuda" for GPU inference.
    trust_remote_code=True,
)

# use it when the flash-attn is available
# model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="auto", attn_implementation='flash_attention_2', trust_remote_code=True)

# forecast
prediction_length = 6
output = model.generate(normed_seqs, max_new_tokens=prediction_length)  # shape is [batch_size, 12 + 6]
normed_predictions = output[:, -prediction_length:]  # shape is [batch_size, 6]
```
