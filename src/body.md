# Introduction
Mobilint® Model Compiler (i.e., Compiler) is a tool that converts models from deep learning frameworks (ONNX, PyTorch, Keras, TensorFlow, etc...) into Mobilint® Model eXeCUtable (i.e., MXQ), a format executable by Mobilint® Neural Processing Unit (NPU). This is the manual for the @<b>qubee@</b>, Mobilint's SDK. In this manual, you can leran how to use it, what kind of frameworks does it support, and etc. A set of functions that can be used to interact with the SDK will be given below.
 
@<img:media/component.svg;0.5;SDK Components>
 
Inputs to qubee are a trained deep learning model, its input shape, and calibration data. It will return MXQ (compiled model) as an output.

@<img:media/qubee.jpg;0.75;Input and output of qubee>

# Changelog
## qubee v0.7.7 (June 2023)
API
    CPU offloading (beta version)
Improve CPU efficiency
Support more operations
Docker
    torch: 1.10.1 -> 1.13.0
    tensorflow: 2.3.0 -> 2.9.0
    onnx:1.11.0 -> 1.12.0

## qubee v0.7 (March 2023)
Multi-channel quantization
Support more operations
API
    Calibration dataset
CPU Offloading (Beta Version)

## qubee v0.6 (August 2022)
Minor updates

## qubee v0.5 (July 2022)
Docker
    Conda -> Virtualenv
    Python: 3.7.7 -> 3.8.10
    torch: 1.8.1 -> 1.10.1
    tensorflow: 1.15.0 -> 2.3.0
    onnx:1.6.0 -> 1.11.0
Parser
    Code refactoring
API
    Enable saving sample inference results (inputs and outputs)

## qubee v0.4 (February 2022)
Optimizer
    Minor updates in fusing reshape

## qubee v0.3 (February 2022)
Parser
    Identify preprocess and postprocess of the model
    Exclude preprocess and postprocess if they are unsupported by the NPU
API
    Simulate integer inference in Python API

## qubee v0.2 (December 2021)
First release

# Installation
## System requirements
We recommend to use NVIDIA GPU for faster compile wtih qubee, but it is nor requirement. Currently, CPU version qubee is also supported.

### Reference System
```vim
Ubuntu 18.04.6 LTS
NVIDIA Graphics Driver 465.19.01
```
 
### Recommended Packages
```vim
NVIDIA Graphics Driver 450.80.02 or Above
Docker
nvidia-docker
```
 
## SDK Installation
We recommend installing qubee on the mobilint docker container. 
(Docker image: mobilint/qbcompiler:v0.7, @<link:https://hub.docker.com/r/mobilint/qbcompiler>)

### Building Docker Image
Run the following commands to build the docker image.
```bash
$ # Docker image download
$ docker pull mobilint/qbcompiler:v0.7
$ # Make a docker container
$ cd {WORKING DIRCTORY}
$ docker run -it --gpus all --name mxq_compiler -v $(pwd):/data mobilint/qbcompiler:v0.7
```
 
### Installation of qubee
Run the following commands to install qubee on the docker container.
```bash
$ # Download qubee-0.7.7-py3-none-any.whl file
$ # Copy qubee whl file to Docker
$ docker cp {Path to qubee-0.7.7-py3-none-any.whl} mxq_compiler:/
$ # Start docker
$ docker start mxq_compiler
$ # Attach docker
$ docker exec -it mxq_compiler /bin/bash
$ # Install qubee
$ cd /
$ python -m pip install qubee-0.7.7-py3-none-any.whl
```

# Tutorials
The tutorials below go through preparing the calibration dataset, model compile and inference steps.

## Preparing Calibration Data
To compile the model, you should prepare the calibration dataset (the pre-processed inputs for the model) for quantization. There are three ways to make the calibration dataset as follows:

(i) Pre-process the raw calibration dataset and save it as numpy tensors.
(ii) Utilize a pre-processing configuration YAML file (only for images).
(iii) Use a manually defined pre-processing function (only for images).

@<b> Important @</b> The process of making a calibration dataset may vary depending on whether you compile the model for CPU offloading or not. Currently, qubee compiles the model without CPU offloading by default. In this scenario, the pre-processed input shape should be in the format (H, W, C). On the other hand, when CPU offloading is employed, the pre-processed input shape should match the input shape that the original model takes.

### Pre-process raw calibration dataset and save it as numpy tensors
You can save the pre-processed calibration dataset as numpy tensors and use them to compile the model. An example code is shown below. The following code assumes that images to be used for calibration is prepared in directory `/workspace/cali_imagenet`.

```python
import os
import numpy as np
import cv2

def get_img_paths_from_dir(dir_path: str, img_ext = ["jpg", "jpeg", "png"]):
    assert os.path.exists(dir_path)
    candidates = os.listdir(dir_path)
    return [os.path.join(dir_path, y) for y in candidates if any([y.lower().endswith(e) for e in img_ext])]

def pre_process(img_path: str, target_h: int, target_w: int):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, dsize=(target_w, target_h)).astype(np.float32)
    return resized_img

if __name__ == "__main__":
    img_dir = "/workspace/cali_imagenet"
    save_dir = "/workspace/calibration/custom_single_input"
    target_h, target_w = 32, 32

    os.makedirs(save_dir, exist_ok=True)
    img_paths = get_img_paths_from_dir(img_dir)
    for i, img_path in enumerate(img_paths):
        fname = f"{i}".zfill(3) + ".npy"
        fpath = os.path.join(save_dir, fname)
        x = pre_process(img_path, target_h, target_w)
        np.save(fpath, x)
```
 
The above results in a directory containing the pre-processed calibration dataset (numpy tensors), located at "/workspace/calibration/custom_single_input".

### Use a pre-processing configuration YAML file
Image pre-processing techniques such as resizing, cropping, and normalization are often applied in machine vision tasks. Users can construct a pre-processing configuration by making use of a YAML file and prepare the calibration dataset via the API provided by qubee, @<i>make_calib@</i>. Please be aware that this method can only be employed when the raw data is in the form of an image. An example code is shown below. The following code assumes that images to be used for calibration is prepared in directory `/workspace/cali_imagenet`.

```python
from qubee import make_calib
make_calib(
    args_pre="mobilenet_v2.yaml", # path to pre-processing configuration yaml file
    data_dir="/workspace/cali_imagenet", # path to folder of original calibration data files such as images
    save_dir="/workspace/sample/calibration", # path to folder to save pre-proceessed calibration data files
    save_name="mobilenet_v2", # tag for the generated calibration dataset
    max_size=50 # Maximum number of data to use for calibration
)
```

```yaml
# mobilenet_v2.yaml
Datatype: Image
GetImage:
    to_float32: false
    channel_order: RGB

Pre-Order: [ResizeTorch, CenterCrop, Normalize, SetOrder]
Pre-processing:
    ResizeTorch:
        size: [256, 256]
        interpolation: bilinear
    CenterCrop:
        size: [224, 224]
    Normalize:
        mean: [0.485, 0.456, 0.406]
        std: [0.229, 0.224, 0.225]
        to_float_div255: true
    SetOrder:
        shape: HWC
```
 
The above results in a directory containing the pre-processed calibration dataset (numpy tensors), located at "/workspace/sample/calibration/mobilenet_v2". Additionally, a calibration meta txt file containing the paths to the pre-processed numpy files is created, named "/workspace/sample/calibration/mobilenet_v2.txt".

### Use a manually defined pre-processing function
You can use your own pre-processing function to make the calibration dataset via the API provided by qubee, @<i>make_calib_man@</i>. In this case, the pre-processing function should take the image path as input and return a numpy tensor. An example of the code is shown below. The following code assumes that images to be used for calibration is prepared in directory `/workspace/cali_imagenet`.

```python
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
from qubee import make_calib_man

def preprocess_resnet50(img_path: str):
    img = Image.open(img_path)
    resize_size=(232, 232)
    crop_size=(224, 224)
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    out = F.pil_to_tensor(img)
    out = F.resize(out, size=resize_size)
    out = F.center_crop(out, output_size=crop_size)
    out = out.to(torch.float, copy=False) / 255.
    out = F.normalize(out, mean, std)
    out = np.transpose(out.numpy(), axes=[1, 2, 0])
    return out

make_calib_man(
    pre_ftn=preprocess_resnet50,
    data_dir="/workspace/cali_imagenet",
    save_dir="/workspace/sample/calibration",
    save_name="resnet50",
    max_size=50
)
```
 
The above results in a directory containing the pre-processed calibration dataset (numpy tensors), located at "/workspace/sample/calibration/resnet50". Additionally, a calibration meta txt file containing the paths to the pre-processed numpy files is created, named "/workspace/sample/calibration/resnet50.txt".

## Compiling ONNX Models
ONNX model can be compiled in two different ways. The first approach involves directly parsing the ONNX model to obtain Mobilint IR. The second approach involves converting the ONNX model to TVM, which is then further converted into Mobilint IR. Once the model is converted to Mobilint IR, then it is be compiled into MXQ. Examples of the code are shown below. The following codes assume that the calibration dataset and the model are prepared in directory `/workspace/cali_imagenet` and `/workspace/resnet18.onnx`, respectively.

```python
""" Compile ONNX model, first way""" 
onnx_model_path = "/workspace/resnet18.onnx"
calib_data_path = "/workspace/cali_imagenet"
# calib_data_path can be replaced with the path to the calibration meta file such as "/workspace/cali_imagenet.txt"

mxq_compile(
    model=onnx_model_path,
    model_nickname="resnet18",
    calib_data_path=calib_data_path,
    backend="onnx"
)
```
 
```python
""" Compile ONNX model, second way """ 
from qubee import mxq_compile

onnx_model_path = "/workspace/resnet18.onnx"
calib_data_path = "/workspace/cali_imagenet"
# A calibration meta file such as "/workspace/cali_imagenet.txt" can be used instead.

mxq_compile(
    model=onnx_model_path,
    model_nickname="resnet18",
    calib_data_path=calib_data_path,
    backend="tvm"
)
```
 
## Compiling PyTorch Models
PyTorch models can be compiled in two different ways. The first approach involves converting the PyTorch model to ONNX, which is then further converted into Mobilint IR. The second approach involves converting the PyTorch model to TVM, which is then further converted into Mobilint IR. Once the model is converted to Mobilint IR, then it is be compiled into MXQ. Examples of the code are shown below. The following codes assume that the calibration dataset is prepared in directory `/workspace/cali_imagenet`.

```python
""" Compile PyTorch model, first way """
from qubee.utils import convert_pytorch_to_onnx 
import torchvision 

input_shape = (224, 224, 3) 
calib_data_path = "/workspace/cali_imagenet"
# A calibration meta file such as "/workspace/cali_imagenet.txt" can be used instead.

### get resnet18 from torchvision and convert it to ONNX 
torch_model = torchvision.models.resnet18(pretrained=True) 
onnx_model_path = "/workspace/resnet18.onnx"
convert_pytorch_to_onnx(torch_model, input_shape, onnx_model_path) 

mxq_compile(
    model=onnx_model_path,
    model_nickname="resnet18",
    calib_data_path=calib_data_path,
    backend="onnx"
)
```

```python
### get resnet18 from torchvision 
import torchvision
torch_model = torchvision.models.resnet18(pretrained=True) 
calib_data_path = "/workspace/cali_imagenet"
# A calibration meta file such as "/workspace/cali_imagenet.txt" can be used instead.

mxq_compile(
    model=torch_model,
    model_nickname="resnet18",
    calib_data_path=calib_data_path,
    backend="tvm",
    input_shape=(224, 224, 3)
)
```
 
## Compiling Keras Models
Keras models are converted to TVM, and further converted into Mobilint IR. Once the model is converted to Mobilint IR, then it is be compiled into MXQ. An example of the code is shown below. The following code assumes that the calibration dataset is prepared in directory `/workspace/cali_imagenet`.

```python
""" Compile Keras model """ 
import tensorflow.keras as keras 

keras_model = keras.applications.resnet18.ResNet18() 
input_shape = (224, 224, 3) 
calib_data_path = "/workspace/cali_imagenet"
# A calibration meta file such as "/workspace/cali_imagenet.txt" can be used instead.

mxq_compile(
    model=keras_model,
    model_nickname="resnet18",
    calib_data_path=calib_data_path,
    backend="tvm",
    input_shape=(224, 224, 3)
)
```
 
## Compiling TensorFlow Models
qubee supports TensorFlow up to version 1.15. So, it requires a frozen TensorFlow PB graph as input, which will be parsed and converted to Mobilint IR. Once the model is converted to Mobilint IR, then it is be compiled into MXQ. An example of the code is shown below. The following code assumes that the calibration dataset is prepared in directory `/workspace/cali_imagenet`.
 
```python
""" Compile Tensorflow model """ 
import wget
import os

### download tensorflow resnet50 from zenodo website 
tf_model = 'resnet50_v1.pb' 
if os.path.isfile(tf_model):
    print('Found cached model: {}'.format(tf_model)) 
else:
    print('Downloading model: {}'.format(tf_model)) 
tf_model = wget.download('https://zenodo.org/record/2535873/files/resnet50_v1.pb') 

input_shape = (224, 224, 3) 
calib_data_path = "/workspace/cali_imagenet"
# A calibration meta file such as "/workspace/cali_imagenet.txt" can be used instead.
mxq_compile(
    model=tf_model,
    model_nickname="resnet50",
    calib_data_path=calib_data_path,
    backend="tf",
    input_shape=(224, 224, 3)
)
```
 
# CPU Offloading
From qubee v0.7, we provide Beta version of CPU offloading for mxq compile. CPU offloading makes it easier for users to compile their models by automatically offloading the computation that are not supported by Mobilint NPU to the CPU. For example, if a pre-processing or post-processing included in the model involves operations that are not supported by the NPU, the user would have to implement them manually after compile, but CPU offloading covers most of these operations and eliminates the need for additional work.

When CPU offloading is employed, the procedures for preparing the calibration dataset and compiling the model vary slightly as follows: 
(i) The pre-processed input shape should match the input shape that the original model takes, whereas the pre-processed input shape should be in the format (H, W, C) to compile the model without CPU offloading. 
(ii) Set the argument @<i>cpu_offload@</i> of function @<i>mxq_compile@</i> True to enable CPU offloading.

@<img:media/offloading_fig.svg;0.85;SDK CPU Offloading>
 
# Supported Frameworks
 We support almost all the commonly used Machine Learning frameworks & libraries such as ONNX, TVM, PyTorch, Keras, and TensorFlow.

@<img:#media/supported_frameworks.png;1.0;Supported deep-learning frameworks>

## Supported Operations (ONNX)
@<tbl:media/supported_onnx.xlsx;Sheet1;ONNX Supported Operations>
 
## Supported operations (PyTorch)
@<tbl:media/supported_pytorch.xlsx;Sheet1;PyTorch Supported Operations>
 
## Supported operations (TensorFlow)
@<tbl:media/supported_tf.xlsx;Sheet1;TensorFlow Supported Operations>
 
## Supported operations (Keras)
@<tbl:media/supported_keras.xlsx;Sheet1;Keras Supported Operations>
 
# API Reference
## Class: Model_Dict
This class serves two main functions:
1. Compile
2. Inference (Note that this inference is only for testing and done by CPU or GPU.)

@<tbl:media/class_model_dict.xlsx;Sheet1;Model_Dict Class>
 
### Methods
@<tbl:media/method_details.xlsx;Sheet1;Model_Dict Methods>
  
### Method Details
@<tbl:media/model_dict.__init__.xlsx;Sheet1;Model_Dict.__init__>
 
@<tbl:media/model_dict.compile.xlsx;Sheet1;Model_Dict.compile>
 
@<tbl:media/model_dict.inference.xlsx;Sheet1;Model_Dict.inference>
 
@<tbl:media/model_dict.inference_int8.xlsx;Sheet1;Model_Dict.inference_int8>
 
@<tbl:media/model_dict.inference_int8_input_dict.xlsx;Sheet1;Model_Dict.inference_int8_input_dict>
 
@<tbl:media/model_dict.to.xlsx;Sheet1;Model_Dict.to>
 

## Function: mxq_compile
Compile a given model directly without creating an instance of "Model_Dict".
@<tbl:media/mxq_compile.xlsx;Sheet1;mxq_compile>
 
## Function: make_calib
From given images and preprocessing configuration, create the preprocessed numpy files and a txt file containing their paths.
@<tbl:media/make_calib.xlsx;Sheet1;make_calib>
 
## Fuction: make_calib_man
From given images and manually written function that takes an image path as input, create the preprocessed numpy files and a txt file containing their paths.
@<tbl:media/make_calib_man.xlsx;Sheet1;make_calib_man>
 
For example, you can use this as follows:
 
```python
import cv2
import numpy as np
from qubee import make_calib_man

def pre_ftn(img_path: str):
    x = cv2.imread(img_path, cv2.IMREAD_COLOR)
    x = x.astype(np.float32) / 255.
    x -= 0.01
    x *= 1.3
    return x

make_calib_man(
    pre_ftn=pre_ftn,
    save_dir="calibration",
    data_dir="/workspace/datasets/color_samples",
    save_name="color_test")
```
 
The above results in a directory containing the pre-processed calibration dataset, located at "/workspace/datasets/color_samples".

## Pre-processing Configurations
qubee supports the following pre-processing functions to make calibration data.
@<tbl:media/pre_process.xlsx;Sheet1;Pre-processing function API>
 
You can write a yaml file as follows:
```yaml
[Pre-processing Type]
    [Parameter]: [Argument]
    ...
```
 
```yaml
# Example
GetImage: 
    to_float32: false
    channel_order: RGB
ResizeTorch: 
    size: [256, 256]
    interpolation: blinear
CenterCrop:
    size: [224, 224]
Normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
    to_float: true
SetOrder:
    shape: HWC
```
 
### Pre-processing Parameters
@<tbl:media/pre_process.GetImage.xlsx;Sheet1;GetImage>
 
@<tbl:media/pre_process.Pad.xlsx;Sheet1;Padding>
 
@<tbl:media/pre_process.Normalize.xlsx;Sheet1;Normalize>
 
@<tbl:media/pre_process.ResizeTorch.xlsx;Sheet1;ResizeTorch>
 
@<tbl:media/pre_process.Resize.xlsx;Sheet1;Resize>
 
@<tbl:media/pre_process.CenterCrop.xlsx;Sheet1;CenterCrop>
 
@<tbl:media/pre_process.SetOrder.xlsx;Sheet1;SetOrder>
 

# Open Source License Notice
@<b>Apache TVM@</b>
* @<link:https://github.com/apache/tvm;https://github.com/apache/tvm>
* Apache 2.0 License
 

@<b>PyTorch@</b>
* @<link:https://github.com/pytorch/pytorch;https://github.com/pytorch/pytorch>
* BSD-like License
 

@<b>TensorFlow@</b>
* @<link:https://github.com/tensorflow/tensorflow;https://github.com/tensorflow/tensorflow>
* Apache 2.0 License
 

@<b>ONNX@</b>
* @<link:https://github.com/onnx/onnx;https://github.com/onnx/onnx>
* Apache 2.0 License
 

@<b>ONNX Runtime@</b>
* @<link:https://github.com/microsoft/onnxruntime;https://github.com/microsoft/onnxruntime>
* MIT License
 

@<b>Keras@</b>
* @<link:https://github.com/keras-team/keras;https://github.com/keras-team/keras>
* Apache 2.0 License

# Copyright
Copyrightⓒ 2019-present, Mobilint, Inc. All rights reserved.