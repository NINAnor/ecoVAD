<h1 align="center">ecoVAD :four_leaf_clover: </h1>
<h2 align="center">An end to end pipeline for training and using VAD models in soundscape analysis.</h2>

![CC BY-NC-SA 4.0][license-badge]
![Supported OS][os-badge]

[license-badge]: https://badgen.net/badge/License/CC-BY-NC-SA%204.0/green
[os-badge]: https://badgen.net/badge/OS/Linux%2C%20Windows/blue

## Introduction

The software is an open source toolkit written in Python for **Voice Active Detection in natural soundscapes and data anonymisation**. It uses our own training pipeline `ecoVAD` which was developped in [PyTorch](https://pytorch.org/) but we also provide wrappers around existing state-of-the-art VAD models to make anonimisation of data more accessible.

Feel free to use ecoVAD for your acoustic analyses and research. If you do, please cite as:

```
Cretois, B., Rosten, C.M. & Sethi, S. S. (2022). Automated speech detection in eco-acoustic data enables privacy protection and human disturbance quantification. bioRxiv.
```

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/

## Repository

This repository contains all the tools necessary to **train** from scratch a deep learning model for Voice Active Detection (VAD) but also to **use** existing ones (namely [pyannote](https://github.com/pyannote/pyannote-audio) and [webrtcvad](https://github.com/wiseman/py-webrtcvad) and our own model trained using the `ecoVAD` pipeline)

## Dataset

If you want to test our pipeline you do not need any dataset, we provided some demo files on OSF at this link: https://osf.io/f4mt5/ so that you can try the pipeline yourself!

:bulb: **Note that the ecoVAD's model weights are also on the OSF folder and you will need to download it if you wish to use our ecoVAD's model.**

Nevertheless, if you want to train a realistic model from scratch you will need your **own soundscape dataset**, a **human speech dataset** (in our analysis we used [LibriSpeech](https://www.openslr.org/12/)) and a **background noise dataset** (in our analysis we used both [ESC50](https://github.com/karolpiczak/ESC-50) or [BirdCLEF](https://www.imageclef.org/lifeclef/2017/bird)). 

## Installation

### Installation without Docker

---

This code has been tested using Ubuntu 18.04 LTS and Windows 10 but should work with other distributions as well. Only Python 3.8 is supported though the code should work with other distributions as well.

1. Clone the repository:

`git clone https://github.com/NINAnor/ecoVAD`

2. Install requirements:

We use [poetry](https://python-poetry.org/) as a package manager which can be installed with the instructions below:

```
cd ecoVAD
pip install poetry 
poetry install --no-root
```

3. Pydub and Librosa require audio backend (FFMPEG)

`sudo apt-get install ffmpeg`

### Installation using Docker

---

First you need to have docker installed on your machine. Please follow the guidelines from the [official documentation](https://docs.docker.com/engine/install/).

1. Clone the repository:

`git clone https://github.com/NINAnor/ecoVAD`

2. Build the image

```
cd ecoVAD
docker build -t ecovad -f Dockerfile .
```

### Download the folder `assets`

---

To be able to run the pipeline with demo data and to get the weights of the model we used in our analysis, it is necessary to download the folder `assets` located on OSF: https://osf.io/f4mt5/.

:arrow_right: Just go to the link, click on `assets.zip` and click on `download`.

Now, simply unzip and place `assets` in the ecoVAD folder.

**You are now set up to run our ecoVAD pipeline!**

## Usage

Our repository provides the necessary scripts and instructions to **train a VAD model** but also to **use existing ones**. If you are only interested in making predictions using an existing model please refer to the section [detecting human speech](#detecting-human-speech).

:bulb: **Note that we recommand using the ecoVAD pipeline if you have a large enough dataset that can be used to train the model, otherwise pyannote is a very good alternative**

Please note that for all the steps below, we provided a Jupyter notebook in `notebooks` so that it is possible to understand and run the scripts step by step.

If you are using your own dataset and wish to have more control over the training and inference pipeline please make sure to change the parameters from the `config_training.yaml` and `confing_inference.yaml`.


### Training your own VAD model using ecoVAD pipeline

---

To generate the synthetic dataset and train the model simply run:

`poetry run python train_ecovad.py`

Or alternatively, if you have docker install and the docker image built:

`docker run --rm -v $PWD/:/app ecovad python train_ecovad.py`


### Detecting human speech using a state-of-the-art VAD model

---

:point_right: **For step you do not need to have trained your own VAD model but you can use existing models**. Just make sure you specify the model you prefer to use in `config_inference.yaml`.

You can run the anonymisation script using:

`poetry run python anonymise_data.py`

Or alternatively, if you have docker install and the docker image built:

`docker run --rm -v $PWD/:/app ecovad python anonymise_data.py`

The anonymisation script will by default output some `.json` files that contains all detections made by the models (the default output folder is `./assets/demo_data_/detections/json/ecoVAD`) that will be used to anonymise the data (which by default are in `./assets/demo_data/anonymised_data`)

### Extracting speech segments

---

You can run the extract segment script using:

`poetry run python extract_detection.py`

Or alternatively, if you have docker install and the docker image built:

`docker run --rm -v $PWD/:/app ecovad python extract_detection.py`

Note that you can choose the number of sampled detections in the `config_inference.yaml`

---

### Contact

If you come across any issues with the ecoVAD pipeline, please open an **issue**.

For other inquiry you can contact me at *benjamin.cretois@nina.no*.



