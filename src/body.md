# Introduction
 Mobilint® Model Compiler (i.e., Compiler) is a tool that converts models from \
deep learning frameworks (ONNX, PyTorch, Keras, TensorFlow, etc...) into Mobilint® \
Model eXeCUtable (i.e., MXQ), a format executable by Mobilint® Neural Processing \
Unit (NPU). This is the manual for the qubee, Mobilint's SDK. In this manual, \
you can leran how to use the SDK, what kind of frameworks does it support, etc. \
A set of functions that can be used to interact with the SDK will be given below.

@<img:#media/sdk_components.png;0.5;SDK Components>

 Input to the SDK is a trained deep learning model, its input shape, and \
calibration data. SDK will return MXQ (compiled model) as an output.

@<img:#media/input_and_output.png;0.75;Input and output of qubee>

# Changelog
## qubee v0.7 (March 2023)
Multi-channel quantization
Support more operations
API to make calibration dataset
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
 In order to use the qubee, the NVIDIA GPU is required. CPU version qubee will \
be provided in the future.

### Reference System
```vim
Ubuntu 18.04.6 LTS
NVIDIA Graphics Driver 465.19.01
```
 

### Requirement Packages
```vim
NVIDIA Graphics Driver 450.80.02 or Above
Docker
nvidia-docker
```
 

## SDK Installation
 We recommend installing qubee on the mobilint docker container. \
(Docker image: mobilint/qbcompiler:v0.4, @<link:https://hub.docker.com/r/mobilint/qbcompiler;https://hub.docker.com/r/mobilint/qbcompiler>)

### Building Docker Image
Run the following commands to build the docker image.
```bash
$ # Docker image download
$ docker pull mobilint/qbcompiler:v0.6
$ # Make a docker container
$ docker run -it --gpus all --name mxq_compiler -v $(pwd):/data mobilint/qbcompiler:v0.6
```


### Installation of qubee
 Run the following commands to install qubee on the docker container.
```bash
$ # Download qubee-0.7-py3-none-any.whl file
$ # Copy qubee-0.7-py3-none-any.whl file to Docker
$ docker cp /path/to/qubee-0.7-py3-none-any.whl mxq_compiler:/
$ # Start docker
$ docker start mxq_compiler
$ # Attach docker
$ # Install qubee
$ cd /
$ python -m pip install qubee-0.7-py3-none-any.whl
```

# Tutorials
 The tutorials below go through preparing calibration dataset, model compile \
and inference steps.

## Preparing Calibration Data
 This step makes calibration data txt file for quantization. This step is \
required before compiling the model.

```python
from qubee import make_calib
args_pre = 'mobilenet_v3_small.yaml' # path to pre-processing configuration yaml file
data_dir = '/mount/datasets/imagenet/mlperf_subset' # path to folder of original calibration data files such as images
save_dir = '/workspace/calibration' # path to folder to save pre-proceessed calibration data files
save_name = 'calib_imagenet_mobilenet_v3' # tag for the generated calibration dataset
max_size = 500 # Maximum number of data to use for calibration
make_calib(args_pre, data_dir, save_dir, save_name, max_size)
```

```yaml
# mobilenet_v3_small.yaml

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

 The above results calibration meta txt file “/workspace/calibration/calib_imagenet_mobilenet_v3.txt”.
 

## Compiling ONNX Models
 ONNX model can be parsed in two different ways. The first one just directly \
parses the ONNX model, converts it to Mobilint IR. The second one converts the \
ONNX model to TVM, parses it, and converts it to Mobilint IR. Once the model is \
converted into Mobilint IR, then it will be compiled into MXQ.
```python
""" Compile ONNX model, first way""" 
onnx_model_path = "/workspace/mount/onnx_models/resnet18.onnx"
calib_txt_path = "/workspace/calibration/calib_imagenet_resnet18.txt"

mxq_compile(
    model=onnx_model_path,
    model_nickname="resnet18",
    calib_txt_path=calib_txt_path,
    backend="onnx"
)
```

```python
""" Compile ONNX model, second way """ 
from qubee import mxq_compile

onnx_model_path = "/workspace/mount/onnx_models/resnet18.onnx"
calib_txt_path = "/workspace/calibration/calib_imagenet_resnet18.txt"

mxq_compile(
    model=onnx_model_path,
    model_nickname="resnet18",
    calib_txt_path=calib_txt_path,
    backend="tvm"
)
```
 

## Compiling PyTorch Models
 PyTorch model can be parsed in two different ways. First, one converts to ONNX, \
parses it, and converts to Mobilint IR. The second one converts to TVM, parses it, \
and converts to Mobilint IR. Once the model is converted to Mobilint IR, then it \
will be compiled into MXQ.

```python
""" Compile PyTorch model, first way """
from qubee.utils import convert_pytorch_to_onnx 
import torchvision 


input_shape = (224, 224, 3) 
calib_txt_path = "/workspace/calibration/calib_imagenet_resnet18.txt"

### get resnet18 from torchvision and convert it to ONNX 
torch_model = torchvision.models.resnet18(pretrained=True) 
onnx_model_path = "/workspace/mount/onnx_models/resnet18.onnx"
convert_pytorch_to_onnx(torch_model, input_shape, onnx_model_path) 

mxq_compile(
    model=onnx_model_path,
    model_nickname="resnet18",
    calib_txt_path=calib_txt_path,
    backend="onnx"
)
```

```python
### get resnet18 from torchvision 
import torchvision
torch_model = torchvision.models.resnet18(pretrained=True) 
calib_txt_path = "/workspace/calibration/calib_imagenet_resnet18.txt"

mxq_compile(
    model=torch_model,
    model_nickname="resnet18",
    calib_txt_path=calib_txt_path,
    backend="tvm",
    input_shape=(224, 224, 3)
)
```
 

## Compiling Keras Models
 Keras model will be to TVM, which will be parsed and converted to Mobilint IR. \
Once the model is converted to Mobilint IR, then it will be compiled into MXQ.

```python
""" Compile Keras model """ 
import tensorflow.keras as keras 

keras_model = keras.applications.resnet18.ResNet18() 
input_shape = (224, 224, 3) 
calib_txt_path = "/workspace/calibration/calib_imagenet_resnet18.txt"

mxq_compile(
    model=keras_model,
    model_nickname="resnet18",
    calib_txt_path=calib_txt_path,
    backend="tvm",
    input_shape=(224, 224, 3)
)
```
 

## Compiling TensorFlow Models
 qubee supports TensorFlow up to version 1.15. So, it requires a frozen \
TensorFlow PB graph as input, which will be parsed and converted to Mobilint IR. \
Once the model is converted to Mobilint IR, then it will be compiled into MXQ.

```python
""" Compile Tensorflow model """
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
calib_txt_path = "/workspace/calibration/calib_imagenet_resnet50.txt"
mxq_compile(
    model=tf_model,
    model_nickname="resnet50",
    calib_txt_path=calib_txt_path,
    backend="tf",
    input_shape=(224, 224, 3)
)
```
# CPU Offloading

From qubee v0.7, we support CPU offloading for mxq compile.
CPU offloading makes it easier for users to compile their models by automatically offloading the computation to the CPU, even if the model contains operations that are not supported by Mobilint NPU.
For example, if a preprocessing or postprocessing function used in deep learning involves operations that are not supported by the NPU, the user would have to implement them manually, but CPU offloading covers most of these operations and eliminates the need for additional work.

# Supported Frameworks
 We support almost all the commonly used Machine Learning frameworks & libraries, \
such as ONNX, TVM, PyTorch, Keras, and TensorFlow.

@<img:#media/supported_frameworks.png;1.0;Supported deep-learning frameworks>

## Supported Operations (ONNX)
@<tbl:media/supported_onnx.xlsx;Sheet1;ONNX Supported Operations>
## Supported operations (PyTorch)
@<tbl:media/supported_pytorch.xlsx;Sheet1;PyTorch Supported Operations>
## Supported operations (TensorFlow)
@<tbl:media/supported_tf.xlsx;Sheet1;TensorFlow Supported Operations>
## Supported operations (Keras)
@<tbl:media/supported_keras.xlsx;Sheet1;Keras Supported Operation>

# API Reference
## Class: Model_Dict
This class serves two main functions:
1. Compile
2. Inference (Note that this inference is done by CPU or GPU.)

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
Make calibration data and a txt file which contains the generated npy file paths given image and pre-processing configuration yaml file.
@<tbl:media/make_calib.xlsx;Sheet1;make_calib>
 
## Pre-processing Configurations
qubee supports the following pre-processing functions to make calibration data
@<tbl:media/pre_process.xlsx;Sheet1;Pre-processing function API>
 
One can write a yaml file as follows:
```yaml
[Pre-processing Type]
    [Parameter key]: [Parameter value]
    ...
```

Example)
```yaml
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
 
@<tbl:media/pre_process.Pad.xlsx;Sheet1;Pad>
 
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