## Semantic Segmentation for Open Vision Computer / Nvidia TX2

This repository contains code needed to train and test a Semantic Segmentation Convolutional Neural Network based on ERFNet.

The core network architecture is based on ERFNet and code here was developed on top of the original repository. If you use this repository in your work, please cite the original paper along with the necessary Open Vision Camera and related papers.

The original repository requires a Python3.6 compatible version of PyTorch. This repository has been made backward compatible with Python3.5 based installations of PyTorch too. If you are using our docker container, you don't need to worry about this.

## Open Vision Computer
https://github.com/osrf/ovc

## ERFNet References
**"Efficient ConvNet for Real-time Semantic Segmentation"**, E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo, IEEE Intelligent Vehicles Symposium    (IV), pp. 1789-1794, Redondo Beach (California, USA), June 2017.
**[Best Student Paper Award]**, [[pdf]](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17iv.pdf)

**"ERFNet: Efficient Residual Factorized ConvNet for Real-time Semantic Segmentation"**, E. Romera, J. M. Alvarez, L. M. Bergasa and R. Arroyo,            Transactions on Intelligent Transportation Systems (T-ITS), **[Accepted paper, to be published in Dec 2017]**. [[pdf]](http://www.robesafe.uah.es/personal/eduardo.romera/pdfs/Romera17tits.pdf)

## Usage
### **Docker Container**:
This image contains all the necessary code and libraries required to train the Semantic Segmentation network on a desktop system. This container has been tested with CUDA 9.0, CuDNN 7.1, Python3.5 and Pytorch v0.4.0

a) First install NVIDIA-DOCKER: https://github.com/NVIDIA/nvidia-docker

b) Pull docker image from Docker-Hub
```
docker pull shreyasskandan/sshreyas-pytorch35
```

c) Launch the docker image
```
sudo nvidia-docker run -it -v /path/to/shared/folder:/data --ipc=host sshreyas-pytorch35
```

### **Training Code**:
a) Edit the training configuration file - *src/ss_segmentation/config/SS_config_train.py*
b) Navigate to the "train" folder of the *ss_segmentation* repository
```
cd /src/ss_segmentation/train/
```
c) Launch training script *SS_train_network.py*
```
python SS_train_network.py
```
If using a multi-GPU machine, such as DGX Station or DGX-1, this docker image doesn't support multi-gpu training, therefore you need to manually specify the device on which to train
```
CUDA_VISIBLE_DEVICES=0 python SS_train_network.py
```
### **Testing Code**:

a) Edit the testing configuration file (currently set up to process a directory of images) - *src/ss_segmentation/config/SS_config_batch_inference.py*
b) Launch the inference script *SS_batch_inference_directory.py*
```
python SS_batch_inference_directory.py
```
If using a multi-GPU machine, such as DGX Station or DGX-1, this docker image doesn't support multi-gpu training, therefore you need to manually specify the device on which to train
```
CUDA_VISIBLE_DEVICES=0 python SS_batch_inference_directory.py
```

### **ROS Node** for real time inference:

a) You will need to build a barebones installation of ROS that uses Python3 (3.5) to be able to subscribe to sensor_msgs::Image and process them using the PyTorch inference script.

You can set up an *independent* ROS install by following these instructions:\
https://gist.github.com/ShreyasSkandan/fd8682253d71c960b2b56376db6bd74a

b) You can proceed to use previous ROS packages as usual. When launching the
ss_segmentation node, make sure to source this ROS environment.\

c) Edit the necessary parameters in semantic_segmentation.launch and set the
required directory location to the directory of your trained model.

d) Launch the ROS Node
```
roslaunch ss_segmentation semantic_segmentation.launch
```

On an NVIDIA TX2 you should see around 120ms inference time for a 640x512
image.\


