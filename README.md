# Neural Micro Services
Deep learning models implemented as web microservices

![NMS Architecture](./assets/nms_architecture.png)


## Running Services in Docker Containers
### Install the NVIDIA Container Toolkit
**Docker containers with CUDA support are recommeded**

(from: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### Configure the production repository:
```bash
$ curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

#### Update the packages list from the repository:
```bash
$ sudo apt-get update
```

#### Install the NVIDIA Container Toolkit packages:
```bash
$ sudo apt-get install -y nvidia-container-toolkit
```

#### Check if CUDA works on Docker (Optional):
```bash
$ sudo docker run --rm --gpus all ubuntu:22.04 nvidia-smi
```

### Builiding the Docker Image
#### Build Docker image for image classifier
```bash
$ cd docker_img_files/image_classifier_server
$ docker build --tag image_classifier .
```

#### Create volumes for persistance 
```bash
$ docker volume create \
--driver local \
-o o=bind \
-o type=none \
-o device="${HOME}/.docker_volues" \
image_classifier_volume
```

#### List available volumes (optional)
```bash
$ docker volume list
```

#### Start Docker image
**NOTE: for the first run, models are downloaded. This can take some time (models are in GBs)**

Run bash

```bash
docker run -it --rm --gpus all -p 8000:8000 -v image_classifier_volume:/root image_classifier bash
```

Run as daemon

```bash
$ docker run -d --gpus all -p 8000:8000 -v image_classifier_volume:/root image_classifier 
```

## Test Services
### Test image classifier service
```bash
$ python image_classifier_test.py --img_path path/to/image
```