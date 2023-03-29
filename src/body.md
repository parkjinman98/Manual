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

* Reference System
```vim
Ubuntu 18.04.6 LTS
NVIDIA Graphics Driver 465.19.01
```

* Requirement packages
```vim
NVIDIA Graphics Driver 450.80.02 or Above
Docker
nvidia-docker
```

## SDK installation
 We recommend installing qubee on the mobilint docker container. \
(Docker image: mobilint/qbcompiler:v0.4, @<link:https://hub.docker.com/r/mobilint/qbcompiler;https://hub.docker.com/r/mobilint/qbcompiler>)

### Building docker image
Run the following commands to build the docker image.
```bash
$ # Docker image download
$ docker pull mobilint/qbcompiler:v0.6
$ # Make a docker container
$ docker run -it --gpus all --name mxq_compiler -v $(pwd):/data mobilint/qbcompiler:v0.6
```

### installation of qubee
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

## Preparing calibration data
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

## Compiling ONNX models
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

## Compiling PyTorch models
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

## Compiling Keras models
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

## Compiling TensorFlow models
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

# Supported Frameworks
 We support almost all the commonly used Machine Learning frameworks & libraries, \
such as ONNX, TVM, PyTorch, Keras, and TensorFlow.

@<img:#media/supported_frameworks.png;1.0;Supported deep-learning frameworks>

## Supported operations (ONNX)
@<tbl:media/supported_onnx.xlsx;Sheet1;ONNX Supported Operations>
## Supported operations (PyTorch)
@<tbl:media/supported_pytorch.xlsx;Sheet1;PyTorch Supported Operations>
## Supported operations (TensorFlow)
@<tbl:media/supported_tf.xlsx;Sheet1;TensorFlow Supported Operations>
## Supported operations (Keras)
@<tbl:media/supported_keras.xlsx;Sheet1;Keras Supported Operation>

# API Reference
## Model_Dict Class
This class serves two main functions: <br>
1. Compile <br>
2. Inference (Note that this inference is done by CPU or GPU testing.) <br>

@<tbl:media/class_model_dict.xlsx;Sheet1;Model_Dict Class>
## Method detail
@<tbl:media/method_details.xlsx;Sheet1;Method detail>
### `__init__`
* Parameters
```python
Model
```
The type should be string or model instance.

Model path or model instance. Model should be instance for the following cases:
Using backend="tvm" and a Keras model
Using backend="tvm" and a PyTorch model

```python
backend
```
The type should be string.

Which framework to use to get the Mobilint IR.
It must be one of "onnx", "tf", and "tvm".
They correspond to deep learning frameworks as follows:
"onnx": ONNX
"tf": TensorFlow
"tvm": PyTorch, Keras, ONNX

```python
input_shape
```
The type should be tuple or list.

Input shape in HWC. Necessary only for using PyTorch model and backend="tvm".

### compile
* Parameters
```python
model_nickname
```
The type should be string.

Model nickname used in qubee. qubee stores previous optimization information to \
recompile same models faster. qubee finds the previously compiled result with nickname.

```python
calib_txt_path
```
The type should be string.

.txt file contain calibration dataset list.
Calibration dataset should be .npy files with np.float32 dtype.

```python
save_path
```
The type should be string.

Filename of the resulting .mxq.
recommend "model_nickname.mxq"

```python
quantize_method
```
The type should be string.
This is optional, and it defaults to percentile.

Quantization method to determine the scale parameter in the quantization.
Now support "Max" or "Percentile"

```python
quantize_percentile
```
The type should be float.
This is optional, and it defaults to 0.9999.

Percentile used for the quantization method "percentile".
Percentile should be between 0 and 1. (Ex. 0.999, 0.9999)

```python
optimization
```
The type should be bool.
This is optional, and it defaults to 5.

If true, it compiles the model with optimization process. If false, qubee uses \
previous optimization information when stored in previous compiling. \
(Nickname should be the same.) It must be set to True on the first compile.

```python
optimization_level
```
The type should be int.

Optimization level in the compiler. If optimization level is high, NPU inference \
could be faster, but it takes more time for compiling. (Recommend: 3~6)

```python
save_sample
```
The type should be bool.

If true, create the "sampleInOut" folder in the current directory and store the \
input and output binary files in it.

### inference
* Parameters
```python
input_tensor
```
The type should be torch.Tensor.

Input tensor with layout BCHW.

### inference_int8
* Parameters
```python
input_tensor
```
The type should be torch.Tensor.

Input tensor with layout BCHW.

### to
* Parameters
```python
device
```
The type should be string.

Target device to use, which must be one of "cpu", "gpu", "cuda".

# Open Source License Notice
@<p>
@<b>Apache TVM@</b>
@<link:https://github.com/apache/tvm;https://github.com/apache/tvm>
Apache 2.0 License@</p>

@<p>
@<b>PyTorch@</b>
@<link:https://github.com/pytorch/pytorch;https://github.com/pytorch/pytorch>
BSD-like License@</p>

@<p>
@<b>TensorFlow@</b>
@<link:https://github.com/tensorflow/tensorflow;https://github.com/tensorflow/tensorflow>
Apache 2.0 License@</p>

@<p>
@<b>ONNX@</b>
@<link:https://github.com/onnx/onnx;https://github.com/onnx/onnx>
Apache 2.0 License@</p>

@<p>
@<b>ONNX Runtime@</b>
@<link:https://github.com/microsoft/onnxruntime;https://github.com/microsoft/onnxruntime>
MIT License@</p>

@<p>
@<b>Keras@</b>
@<link:https://github.com/keras-team/keras;https://github.com/keras-team/keras>
Apache 2.0 License@</p>

# Copyright
Copyrightⓒ 2019-present, Mobilint, Inc. All rights reserved.