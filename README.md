# Grid-based Super-resolution using Spatial Hash Encoding


## Abstract
There are mainly two types of super-resolution methods: traditional methods and deep learning methods. While traditional methods define closed-form expressions with assumptions, deep learning methods rely on priors learned from data sets. However, both of them have disadvantages such as being too simple and having strong trust in priors. We focus on how to generate a high-resolution image using low-resolution images without priors by utilizing spatial hash encoding. We propose a grid-based super-resolution model using spatial hash encoding to map coordinate information into higher dimensional space. Our aim is to eliminate long training times and not rely on priors from data sets that are not able to cover all real-world scenarios. Therefore, our proposed model is able to do task-specific super-resolution without priors and eliminate potential hallucination effects caused by wrong priors.

## Resources
* [Thesis](https://drive.google.com/file/d/1AWfISJWV3Oprw1L4Ocu1nhsFRgt20fF4/view?usp=sharing)
* [Presentation](https://drive.google.com/file/d/19tlgAtd4bDhpMBIG8JVKRxqcm2-hRyp9/view?usp=sharing)

## Requirements
* Docker 24.0.5
* Docker Compose v2.20.2

## Setup
Clone the repository.

```bash
git clone https://github.com/veliglmz/grid-based-super-resolution-using-spatial-hash-encoding.git
cd grid-based-super-resolution-using-spatial-hash-encoding
```

Build the docker image.
```bash
docker compose build
```

Run the docker image. (the outputs are in the results folder of the host.)
```bash
docker compose run app
```

Stop containers and remove containers, networks, volumes, and images.
```bash
docker compose down
```

## For the data generation:
We use [NTIRE22 BURSTSR Dataset Generation](https://github.com/goutamgmb/NTIRE22_BURSTSR) framework with forward and inverse camera pipeline code from [timothybrooks/unprocessing](https://github.com/timothybrooks/unprocessing). \
In the datasets/NTIRE22_BURSTSR_dataset_generation/main.py, there are two options: MFSR and SISR. \
It creates a dataset from original images randomly based on types. MFRS for 4x upsamling and SISR for 8x upsampling.

```bash 
cd datasets/NTIRE22_BURSTSR_dataset_generation
python main.py -t MFSR
```

