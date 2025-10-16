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

## Citation

> Please let us know if you find out a mistake or have any suggestions!

> If you find the MIRA models helpful in your research, please consider to star this repository and cite the
> corresponding [paper](https://arxiv.org/abs/2506.07584):

```
@article{li2025mira,
  title={MIRA: Medical Time Series Foundation Model for Real-World Health Data},
  author={Li, Hao and Deng, Bowen and Xu, Chang and Feng, Zhiyuan and Schlegel, Viktor and Huang, Yu-Hao and Sun, Yizheng and Sun, Jingyuan and Yang, Kailai and Yu, Yiyao and others},
  journal={arXiv preprint arXiv:2506.07584},
  year={2025}
}
```

## Related Resources
* Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts, in *ICLR*. [\[paper\]](https://arxiv.org/abs/2409.16040)
* TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis, in arXiv 2024. [\[paper\]](https://arxiv.org/abs/2410.16032) [\[GitHub Repo\]](https://github.com/kwuking/TimeMixer)
* Towards Neural Scaling Laws for Time Series Foundation Models, arXiv 2024. [\[paper\]](https://arxiv.org/pdf/2410.12360)
* Foundation Models for Time Series Analysis: A Tutorial and Survey, in *KDD*
  2024. [\[paper\]](https://arxiv.org/abs/2403.14735) [\[Tutorial\]](https://wenhaomin.github.io/FM4TS.github.io/)
* What Can Large Language Models Tell Us about Time Series Analysis, in *ICML*
  2024. [\[paper\]](https://arxiv.org/abs/2402.02713)
* Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects, in *TPAMI*
  2024. [\[paper\]](https://arxiv.org/abs/2306.10125) [\[Website\]](https://github.com/qingsongedu/Awesome-SSL4TS)
* Transformers in Time Series: A Survey, in *IJCAI*
  2023. [\[paper\]](https://arxiv.org/abs/2202.07125) [\[GitHub Repo\]](https://github.com/qingsongedu/time-series-transformers-review)
* A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection, in *TPAMI* 2024. [\[paper\]](https://arxiv.org/abs/2307.03759) [\[Website\]](https://github.com/KimMeen/Awesome-GNN4TS)


## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- Time-LLM [\[repo\]](https://github.com/KimMeen/Time-LLM)
- TimeMixer [\[repo\]](https://github.com/kwuking/TimeMixer)
- Time-Series-Library [\[repo\]](https://github.com/thuml/Time-Series-Library)
- Large (Language) Models and Foundation Models (LLM, LM, FM) for Time Series and Spatio-Temporal
  Data [\[repo\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)

## License

This project is licensed under the Apache-2.0 License.

