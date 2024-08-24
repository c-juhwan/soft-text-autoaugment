# [EACL 2024 SRW] AutoAugment Is What You Need: Enhancing Rule-based Augmentation Methods in Low-resource Regimes

## Introduction

This repository contains the source code and datasets for the paper ["AutoAugment Is What You Need: Enhancing Rule-based Augmentation Methods in Low-resource Regimes"](https://aclanthology.org/2024.eacl-srw.1.pdf) accepted at EACL 2024 Student Research Workshop. We propose to optimize [softEDA](https://openreview.net/pdf?id=OiSbJbVWBJT) with AutoAugment to enhance the performance of rule-based augmentation methods in low-resource regimes. We demonstrate that our method can effectively improve the performance of recent PLMs. Please refer to the paper for more details.

## Experiment

Prepare a virtual environment (Python 3.8) and install the requirements.

```shell
$ conda create -n proj-soft-taa python=3.8
$ conda activate proj-soft-taa
$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
$ pip install -r requirements.txt
$ bash run_baseline.sh
$ bash run_softtaa.sh
$ bash run_ablation.sh
```

## Citation

If you found this work helpful for your future research, please consider citing this work:

```bibtex
@inproceedings{choi2024autoaugment,
  title={AutoAugment Is What You Need: Enhancing Rule-based Augmentation Methods in Low-resource Regimes},
  author={Choi, Juhwan and Jin, Kyohoon and Lee, Junho and Song, Sangmin and Kim, Youngbin},
  booktitle={Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: Student Research Workshop},
  pages={1--8},
  year={2024}
}
```
