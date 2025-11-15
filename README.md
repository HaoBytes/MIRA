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
pip install torchdiffeq
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

For distributed training on irregular medical data:

```bash
python torch_dist_run.py main.py \
  --from_scratch \
  -d ./yourdata.jsonl \
  --output_path ./saveyoucheckpoints \
  --save_steps 10000 \
  --save_strategy steps \
  --save_total_limit 10 \
  --save_only_model \
  --precision bf16 \
  --time_aware_dataset \
  --time_aware_rotary
```

For full argument list:

```bash
python main.py --help
```


## Inference

```python
Below is a minimal example of running MIRA for forecasting:

```python
import torch
from MIRA.mira.models.modeling_mira import MIRAForPrediction

model = MIRAForPrediction.from_pretrained("YOUR_CHECKPOINT").cuda()
seq   = torch.randn(1, 40)
time  = torch.cumsum(torch.rand(1, 40) * 0.15 + 0.05, dim=1)

from examples.mira_inference_demo import mira_forecast
pred = mira_forecast(model, seq, time, context_length=12, pred_length=6)
print(pred)
```

## Datasets

> **Note:** All datasets used in this project are clinical or physiological time-series datasets. Because these datasets contain sensitive human subject information, they are governed by strict data-use agreements (DUA) and protected access policies. Therefore, the raw datasets cannot be redistributed in this repository. You must apply for access through the official data providers listed below.

- **MIMIC** — 
Access link: https://physionet.org/content/mimiciv/
- **WAVES Pediatric Waveform Database** — 
Access link: https://redivis.com/WAVES/datasets
- **PTB-XL** — 
Access link: https://physionet.org/content/ptb-xl/1.0.3/
- **Sleep-EDF** — 
Access link: https://physionet.org/content/sleep-edfx/1.0.0/


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


## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- Time-MoE [\[repo\]](https://github.com/Time-MoE/Time-MoE)
- Time-LLM [\[repo\]](https://github.com/KimMeen/Time-LLM)
- TimeMixer [\[repo\]](https://github.com/kwuking/TimeMixer)
- Time-Series-Library [\[repo\]](https://github.com/thuml/Time-Series-Library)
- Large (Language) Models and Foundation Models (LLM, LM, FM) for Time Series and Spatio-Temporal
  Data [\[repo\]](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)

## License

This project is licensed under the MIT License.

