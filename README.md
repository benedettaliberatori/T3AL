<div align="center">

[![Paper]()]()

# Test-Time Zero-Shot Temporal Action Localization

Benedetta Liberatori, [Alessandro Conti](), [Paolo Rota](), [Yiming Wang](https://www.yimingwang.it/), [Elisa Ricci](http://elisaricci.eu/) <br>



</div>

<p align="center">
  <img style="width: 100%" src="/media/method.pdf">
</p>
<br>

> **Abstract:** **

# Setup

We recommend the use of a Linux machine with CUDA compatible GPUs. We provide a Conda environment to configure the required libraries.

Clone the repo with:

```bash
git clone ...
cd T3AL
```

## Conda

The environment can be installed and activated with:

```bash
conda create --name t3al python=3.8
conda activate t3al
pip install -r requirements.txt
```


# Preparing Datasets
We recommend to use pre-extracted CoCa features to accelerate inference. Please download the extracted features for THUMOS14 and ActivityNet datasets from links below: 

## Pre-extracted Features

# Evaluation

The method can be evaluated on the dataset of interest by running the following bash script:

```bash
python src/train.py experiment=tt_<dataset_name> data=<dataset_name>  model.split=0 data.nsplit=0 exp_name=<exp_name>
```

We provide config files for the main method `tt_<dataset_name>`, the training free baseline `tf_<dataset_name>` and the baselines `baseline`. 

# Citation

Please consider citing our paper in your publications if the project helps your research.

```
bibtex
```