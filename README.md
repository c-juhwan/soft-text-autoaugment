# AutoAugment Is What You Need: Enhancing Rule-based Augmentation Methods in Low-resource Regimes

## How to start

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
