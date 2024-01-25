# Semantic Anomaly Segmentation Benchmark

This project aims to provide a standardized evaluation platform for anomaly segmentation methods in the field of computer vision. Anomaly segmentation is a critical task in various applications, and this repository serves as a hub for assessing the performance of different anomaly detection methods. The benchmark supports several state-of-the-art repositories, including:

1. KDAD ([paper](https://arxiv.org/abs/2011.11108) | [original repo](https://github.com/rohban-lab/Knowledge_Distillation_AD))
2. RD4AD ([paper](https://arxiv.org/abs/2201.10703) | [original repo](https://github.com/hq-deng/RD4AD))
3. Puzzle_AD ([paper](https://arxiv.org/pdf/2008.12959.pdf) | [original repo](https://github.com/Niousha12/Puzzle_Anomaly_Detection))
4. Mean-Shifted AD ([paper](https://arxiv.org/pdf/2106.03844.pdf) | [original repo](https://github.com/talreiss/Mean-Shifted-Anomaly-Detection))
5. Transformaly ([paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Cohen_Transformaly_-_Two_Feature_Spaces_Are_Better_Than_One_CVPRW_2022_paper.pdf) | [original repo](https://github.com/MatanCohen1/Transformaly))
6. Deep-SVDD ([paper](http://proceedings.mlr.press/v80/ruff18a.html) | [original repo](https://github.com/lukasruff/Deep-SVDD-PyTorch))

## Supported Datasets
The benchmark currently supports a set of datasets, and efforts are ongoing to include more diverse datasets over time. 

### Current Supported Datasets:
- MNIST
- FMNIST
- CIFAR10
- CIFAR100
- MVTec
- Pascal VOC

*(List will be updated as more datasets are incorporated. Contributions are welcome!)*

## To Be Added
This repository will go under heavy refactoring in the future and there will only be one single env.yml file and one script that calls the preferred method on the dataset of the user's choice.

## Getting Started
To get started with the benchmark, follow the instructions in the respective repository folders. Use the requirements.txt files to create the environment.

## License
This benchmark is distributed under the [MIT License](LICENSE). Please review the license file before using or contributing to the repository.
