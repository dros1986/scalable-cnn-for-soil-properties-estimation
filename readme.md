# Dependencies
```python
pip install pytorch-lightning
pip install "ray[tune]"
```

# Project "Pignoletto"
This project aims to estimate the fertility of a field through the analysis of nir, swir and gamma signals.

[Read wiki for more details.](https://github.com/dros1986/pignoletto/wiki)


# Docker
Install:
1 - docker ([instructions at this link](https://docs.docker.com/engine/install/ubuntu/))
2 - nvidia-container-toolkit:

```bash
# find most suitable distribution
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# install nvidia-docker2 package
sudo apt-get update && sudo apt-get install -y nvidia-docker2 && sudo systemctl restart docker

# test if everything is working
sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

Then, clone the images and compose them:

```bash
# login to docker
docker login -u dros1986
# get ray. Contains also pytorch.
docker pull rayproject/ray-ml:latest-gpu
# to test if everything is ok:
docker run --gpus all -ti rayproject/ray-ml:latest-gpu
```
