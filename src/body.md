# Introduction
Mobilint® Model Compiler (i.e., Compiler) is a tool that converts models from deep learning frameworks (ONNX, PyTorch, Keras, TensorFlow, etc...) into Mobilint® Model eXeCUtable (i.e., MXQ), a format executable by Mobilint® Neural Processing Unit (NPU). This is the manual for the @<b>qubee@</b>, Mobilint's SDK. In this manual, you can learn how to use it, what frameworks it supports, etc. A set of functions you can use to interact with the SDK will be given below.
 
@<img:media/component.svg;0.5;SDK Components>
 
Inputs to qubee are a trained deep learning model, its input shape, and calibration data. It will return MXQ (compiled model) as an output.

@<img:media/qubee.jpg;0.75;Input and output of qubee>

# Changelog

## qubee v0.8.3 (March 2024)

## qubee v0.8.2 (February 2024)

## qubee v0.8.1 (December 2023)

## qubee v0.8.0 (November 2023)
API
    TVM backend deprecated

## qubee v0.7.12 (September 2023)

## qubee v0.7.11 (August 2023)
API
    Support TorchScript backend

## qubee v0.7.10 (August 2023)

## qubee v0.7.9 (August 2023)

## qubee v0.7.8 (August 2023)

## qubee v0.7.7 (June 2023)
API
    Improve CPU offloading (beta version)
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
    Improve calibration dataset processing
    Support CPU offloading (beta version)

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
We recommend to use NVIDIA GPU for faster compile wtih qubee, but it is not necessary. Currently, CPU version qubee is also supported.

### Reference System
```vim
Ubuntu 22.04.4 LTS
NVIDIA Graphics Driver 545.29.06
```
 
### Requirements and Recommended Packages
```vim
Ubuntu 20.04.6 LTS or Above
NVIDIA Graphics Driver 450.80.02 or Above
Docker
nvidia-docker
```
 
## SDK Installation
We recommend installing qubee on the Mobilint docker container. 
(Docker image: mobilint/qbcompiler:v0.8, @<link:https://hub.docker.com/r/mobilint/qbcompiler>)

### Building Docker Container
Run the following commands to build the docker container.
```bash
$ # Download Docker Image
$ docker pull mobilint/qbcompiler:v0.8
$ # mkdir {WORKING DIRCTORY} (if needed)
$ cd {WORKING DIRCTORY}
$ docker run -it --gpus all --ipc=host --name {YOUR_CONTAINER_NAME} -v $(pwd):/workspace mobilint/qbcompiler:v0.8 /bin/bash
```
(Recommended) If the trained models and datasets are stored in different directories, you can mount them to the docker container as follows:
```bash
$ docker run -it --gpus all --ipc=host --name {YOUR_CONTAINER_NAME} -v $(pwd):/workspace -v {PATH TO MODEL DIR}:/models -v {PATH TO DATASET DIR}:/datasets mobilint/qbcompiler:v0.8 /bin/bash
```
(Optional) Build the docker image for CPU only version
```bash
$ # Download Docker Image
$ docker pull mobilint/qbcompiler:v0.8-cpu
$ cd {WORKING DIRCTORY}
$ docker run -it --ipc=host --name {YOUR_CONTAINER_NAME} -v $(pwd):/workspace mobilint/qbcompiler:v0.8-cpu /bin/bash
```


(Optional, the latest version is not available yet) Build the docker image for WSL2
```bash
$ # Download Docker Image
$ docker pull mobilint/qbcompiler:v0.7-wsl
$ # Make a docker container
$ cd {WORKING DIRCTORY}
$ docker run -it --gpus all --ipc=host --name {YOUR_CONTAINER_NAME} -v $(pwd):/data mobilint/qbcompiler:v0.7-wsl
```

### Installation of qubee
qubee compiler packages are available in @<link:https://dl.mobilint.com/view.php ;Mobilint® Software Development Kit (SDK)>.

Run the following commands to install qubee on the docker container.
```bash
$ # Download qubee-0.8.3-py3-none-any.whl file
$ # Copy qubee whl file to Docker
$ docker cp {Path to qubee-0.8.3-py3-none-any.whl} {YOUR_CONTAINER_NAME}:/
$ # Start Docker
$ docker start {YOUR_CONTAINER_NAME}
$ # Attach to Docker container
$ docker exec -it {YOUR_CONTAINER_NAME} /bin/bash
$ # Install qubee compiler
$ cd /
$ python -m pip install qubee-0.8.3-py3-none-any.whl
```

(Option, for WSL2) Run the following commands to install qubee on the docker container.
```bash
$ # Download qubee-0.8.1_wsl-py3-none-any.whl file
$ # Copy qubee whl file to Docker
$ docker cp {Path to qubee-0.8.1_wsl-py3-none-any.whl} {YOUR_CONTAINER_NAME}:/
$ # Start Docker
$ docker start {YOUR_CONTAINER_NAME}
$ # Attach to Docker container
$ docker exec -it {YOUR_CONTAINER_NAME} /bin/bash
$ # Install qubee
$ cd /
$ python -m pip install qubee-0.8.1_wsl-py3-none-any.whl
```
# Tutorials
The tutorials below go through the steps for preparing the calibration dataset, model compiling, and inference.

## Preparing Calibration Data
To compile the model, you should prepare the calibration dataset (the pre-processed inputs for the model) for quantization. There are three ways to make the calibration dataset as follows:

(i) Pre-process the raw calibration dataset and save it as numpy tensors.
(ii) Utilize a pre-processing configuration YAML file (only for images with @<b>uniform format@</b>).
(iii) Use a manually defined pre-processing function (only for images with @<b>uniform format@</b>).
(iv) Use @<link:https://git.mobilint.com/algorithm-team/utility/calibration_gui;Mobilint® Calibration GUI Tool>

@<b> Important @</b> The process of making a calibration dataset may vary depending on whether you compile the model for CPU offloading or not. Currently, qubee compiles the model without CPU offloading by default. In this scenario, the pre-processed input shape should be in the format (H, W, C). On the other hand, when CPU offloading is employed, the pre-processed input shape should match the input shape that the original model takes.

### Pre-process raw calibration dataset and save it as numpy tensors
You can save the pre-processed calibration dataset as numpy tensors with your custom pre-processing function and use them to compile the model. 

An example code is shown below. The following code assumes that we have an image folder consisting of 1000 randomly selected JPEG image files from the @<link:https://www.image-net.org/;ImageNet> dataset for calibration prepared in directory `/datasets/imagenet/cali_1000`.

```python
import os
import numpy as np
import cv2

def get_img_paths_from_dir(dir_path: str, img_ext = ["jpg", "jpeg", "png"]):
    assert os.path.exists(dir_path)
    candidates = os.listdir(dir_path)
    return [os.path.join(dir_path, y) for y in candidates if any([y.lower().endswith(e) for e in
    img_ext])]
    
def pre_process(img_path: str, target_h: int, target_w: int):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    resized_img = cv2.resize(img, dsize=(target_w, target_h)).astype(np.float32)
    return resized_img

if __name__ == "__main__":
    img_dir = "/datasets/imagenet/cali_1000"
    save_dir = "/workspace/calibration/custom_single_input"
    target_h, target_w = 224, 224
    os.makedirs(save_dir, exist_ok=True)
    img_paths = get_img_paths_from_dir(img_dir)
    for i, img_path in enumerate(img_paths):
        fname = f"{i}".zfill(3) + ".npy"
        fpath = os.path.join(save_dir, fname)
        x = pre_process(img_path, target_h, target_w)
        np.save(fpath, x)
```
 
The above results are in a directory containing the pre-processed calibration dataset (numpy tensors of shape (224,224, 3)), located at `/workspace/calibration/custom_single_input`.

### Use a pre-processing configuration YAML file
Image pre-processing techniques such as resizing, cropping, and normalization are often applied in machine vision tasks. Users can construct a pre-processing configuration using a YAML file and prepare the calibration dataset via the API provided by qubee, @<i>make_calib@</i>. Please be aware that this method can only be employed when the raw data is an image. An example code is shown below. The following code assumes that images for calibration are prepared in the directory `/workspace/cali_1000`.

```python
from qubee import make_calib
make_calib(
    args_pre="/workspace/mobilenet_v2.yaml", # path to pre-processing configuration yaml file
    data_dir="/datasets/imagenet/cali_1000", # path to folder of original calibration data files such as images
    save_dir="/workspace/calibration/", # path to folder to save pre-proceessed calibration data files
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

The above results are in a directory containing the pre-processed calibration dataset (numpy tensors), located at `/workspace/calibration/mobilenet_v2`. In addition, a calibration meta txt file containing the paths to the pre-processed numpy files is created, named `/workspace/calibration/mobilenet_v2.txt`.

@<b>Remark@</b> The sample dataset for calibration should be composed of images with the same format. If some are in color images and others are in grayscale images, the calibration dataset will not be created properly.

### Use a manually defined pre-processing function
You can use your pre-processing function to make the calibration dataset via the API provided by qubee, @<i>make_calib_man@</i>. In this case, the pre-processing function should take the image path as input and return a numpy tensor. An example of the code is shown below. The following code assumes that images for calibration are prepared in the directory `/datasets/imagenet/cali_1000`.

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
    pre_ftn=preprocess_resnet50, # callable function to pre-process the calibration data
    data_dir="/datasets/imagenet/cali_1000", # path to folder of original calibration data files such as images
    save_dir="/workspace/calibration/", # path to folder to save pre-proceessed calibration data files
    save_name="resnet50", # tag for the generated calibration dataset
    max_size=50 # Maximum number of data to use for calibration
)
```

The above results are in a directory containing the pre-processed calibration dataset (numpy tensors), located at `/workspace/sample/calibration/resnet50`. In addition, a calibration meta txt file containing the paths to the pre-processed numpy files is created, named `/workspace/sample/calibration/resnet50.txt`.

@<b>Remark@</b> Unless the custom pre-processing function contains proper exception handling, the sample dataset for calibration should be composed of images with the same format. Like the previous method, the calibration dataset will not be created properly if some are in color images and others are in grayscale images.

### Use Mobilint® Calibration GUI Tool
@<link:http://git.mobilint/AlgorithmGroup/Calibration_GUI; Mobilint® Calibration GUI> is a tool that helps users to make the calibration dataset. With a prepared dataset of image files, users can easily generate a pre-processed calibration dataset of npy files. The tool provides pre-defined pre-processing functions for various deep learning models.

@<img:media/mobilint_calibration_gui.jpg;0.75; Mobilint® Calibration GUI>

## Compiling ONNX Models
ONNX is a recommended framework to be used for compiling the trained model. With simple code, the ONNX model can be directly parsed to obtain Mobilint IR.  example code is shown below. The following code assumes that the calibration dataset and the model are prepared in the directory `/workspace/calibration/resnet50` and `/workspace/resnet50.onnx`, respectively.

```python
""" Compile ONNX model""" 
from qubee import mxq_compile
onnx_model_path = "/workspace/resnet50.onnx"
calib_data_path = "/workspace/calibration/resnet50"
# calib_data_path can be replaced with the path to the calibration meta file such as "/workspace/calibration/resnet50.txt"

mxq_compile(
    model=onnx_model_path,
    calib_data_path=calib_data_path,
    save_path="resnet50.mxq",
    backend="onnx"
)
```
 
## Compiling PyTorch Models
PyTorch models can be compiled in two different ways. The first approach is converting the PyTorch model into the ONNX model with @<link:https://pytorch.org/docs/stable/onnx.html;`torch.onnx`> namespace, and compiling the converted model with the ONNX backend. The second approach is directly plugging the model into the Mobilint IR. Once the model is converted to Mobilint IR, then it is compiled into MXQ. The example code is shown below. The following codes assume that the calibration dataset is prepared in the directory `/workspace/calibration/resnet50`.

```python
""" Compile PyTorch model"""
from qubee import mxq_compile
### get resnet50 from torchvision 
import torchvision
import torch
calib_data_path = "/workspace/calibration/resnet50"
# A calibration meta file such as "/workspace/calibration/resnet50.txt" can be used instead.

### get resnet50 from torchvision and convert it to torchscript
torch_model = torchvision.models.resnet50(pretrained=True) 
torchscript_model_path = "/workspace/resnet50.pt"

example_input = torch.rand(1, 3, 224, 224)
scripted_model = torch.jit.script(torch_model, example_input)
torch.jit.save(scripted_model, torchscript_model_path)

mxq_compile(
    model=torchscript_model_path,
    calib_data_path=calib_data_path,
    backend="torchscript",
    save_path="resnet50.mxq",
    example_input=example_input
)
```

## Compiling TensorFlow/Keras Models
Since Keras works as an interface for TensorFlow, models on the Keras framework can be converted to Mobilint IR via TensorFlow. First, we load and save the Keras/TensorFlow model into the format of the frozen graph, which ends with `.pb`. Then, with the directory containing the frozen graph, qubee will compile the model. The following code assumes the calibration dataset is prepared in the directory `/workspace/calibration/resnet50`.

@<b>@<color:FF0000>Important@</color:FF0000>@</b> According to the annotations and old version instructions, the TensorFlow compilation should work by providing the directory containing the frozen graph or just the frozen graph file. However, the current version of qubee has some bugs in the TensorFlow parser. It is now fixed and will be released in the next version. For now, please use the ONNX or PyTorch model to compile the model.
```python
""" Compile Tensorflow Lite model """ 
from qubee import mxq_compile
import tensorflow as tf

keras_model = tf.keras.applications.resnet50.ResNet50() # Load a pretrained Keras model
input_shape = (224, 224, 3) 
calib_data_path = "/workspace/calibration/resnet50"
# A calibration meta file such as "/workspace/calibration/resnet50.txt" can be used instead.

keras_model_save_path = "/workspace/tf_models/resnet50" # directory to save the Tensorflow model
keras_model.save(keras_model_save_path) # Save the model in the format of. saved_model.pb file, which will be created in the directory.

tflite_model = tf.lite.TFLiteConverter.from_saved_model(keras_model_save_path).convert()
with open('/workspace/tf_models/resnet50.tflite', 'wb') as f:
    f.write(tflite_model)

mxq_compile(
    model=keras_model_save_path+".tflite",
    calib_data_path=calib_data_path,
    backend="tflite",
    save_path="resnet50.mxq",
)
```

## Compiling TensorFlow Lite Models
The qubee compiler supports TensorFlow Lite models. With the given TensorFlow Lite model, the calibration dataset, and the backend, the model can be compiled into Mobilint IR. The following code assumes the calibration dataset is prepared in the directory `/workspace/calibration/resnet50`.

@<b>@<color:FF0000>Important@</color:FF0000>@</b> Currently, the TensorFlow Lite model is not supported in the qubee compiler. Please use the ONNX or PyTorch model to compile the model.

```python
""" Compile Tensorflow Lite model """ 
from qubee import mxq_compile
import tensorflow as tf

keras_model = tf.keras.applications.resnet50.ResNet50() # Load a pretrained Keras model
input_shape = (224, 224, 3) 
calib_data_path = "/workspace/calibration/resnet50"
# A calibration meta file such as "/workspace/calibration/resnet50.txt" can be used instead.

keras_model_save_path = "/workspace/tf_models/resnet50" # directory to save the Tensorflow model
keras_model.save(keras_model_save_path) # Save the model in the format of. saved_model.pb file, which will be created in the directory.

tflite_model = tf.lite.TFLiteConverter.from_saved_model(keras_model_save_path).convert()
with open('/workspace/tf_models/resnet50.tflite', 'wb') as f:
    f.write(tflite_model)

mxq_compile(
    model=keras_model_save_path+".tflite",
    calib_data_path=calib_data_path,
    backend="tflite",
    save_path="resnet50.mxq",
)
```

## Compling Models with Custom Input

When the model lacks input shape information, qubee may generate the following error:
```bash
ValueError: Input node <node name> has more than one unknown shape. Please enter the numpy input array to infer the input shape.
```
If you encounter this error, you should provide numpy input arrays along with the model during compilation. Ensure the folder structure is as follows:
```bash
- <folder name>
|- <input node name 1>.npy // only for input node whose shape is unknown.
|- ...
|- <input node name n>.npy
```
For example, if your model has three inputs named `<input1, input2, input3>` and the shape of `input2` and `input3` are unknown, then you should prepare the numpy array for `input2` and `input3` with the following folder structure.

```bash
- custom_input_array
|- input2.npy
|- input3.npy
```

With the above array, you can compile the model as follows:
```python
# compile_test.py
import argparse
 
from qubee import mxq_compile
from qubee.utils.utils_model_dict import parse_custom_input_info
 
onnx_model_path = "/workspace/deeplabv3_mobilenet_v3_large_torchvision.onnx"
calib_data_path = "/workspace/calibration/deeplabv3"
 
parser = argparse.ArgumentParser(description="Compile arguments")
parser.add_argument("--input_shape_path", dest="custom_input_shape_dict", action=parse_custom_input_info)
 
 
mxq_compile(
    model=onnx_model_path,
    calib_data_path=calib_data_path,
    save_path="deeplabv3.mxq",
    input_shape_dict=args.custom_input_shape_dict
    backend="onnx"
)
```

Then, you can compile the model with the following command:
```bash
python compile_test.py --input_shape_path /workspace/custom_input_array
```

# CPU Offloading (Beta Version)

@<b>Remark@</b> To proceed inference with CPU offloading, it requires a runtime library that supports the MXQ file that is compiled for CPU offloading.

From qubee v0.7, we provide a Beta version of CPU offloading for mxq compile. CPU offloading makes it easier for users to compile their models by automatically offloading the computation that Mobilint NPU does not support to the CPU. For example, if a pre-processing or post-processing included in the model involves operations that the NPU does not support, the user would have to implement them manually after compiling, but CPU offloading covers most of these operations and eliminates the need for additional work.

When CPU offloading is employed, the procedures for preparing the calibration dataset and compiling the model vary slightly as follows: 
(i) The pre-processed input shape should match the original model's input shape, whereas the pre-processed input shape should be in the format (H, W, C) to compile the model without CPU offloading. 
(ii) Set the argument @<i>cpu_offload@</i> of function @<i>mxq_compile@</i> True to enable CPU offloading.

@<img:media/offloading_fig.svg;0.75;SDK CPU Offloading>
 
# Supported Frameworks
We support almost all the commonly used Machine Learning frameworks & libraries such as ONNX, PyTorch, Keras, TensorFlow, and TensorFlow Lite.

@<img:#media/supported_frameworks.png;1.0;Supported deep-learning frameworks>

## Supported Operations (ONNX)
@<tbl:media/supported_onnx.xlsx;Sheet1;ONNX Supported Operations>
 
## Supported operations (PyTorch)

@<b>Remark@</b> Since the Torchscript backend framework is based on @<link:https://pytorch.org/docs/stable/onnx.html#Torchscript-Based-ONNX-Exporter;Torchscript-Based-ONNX-Exporter>, even if the operation is not listed below, it may be supported if it has corresponding ONNX operation, which is supported by qubee.

@<tbl:media/supported_pytorch.xlsx;Sheet1;PyTorch Supported Operations>
 
## Supported operations (TensorFlow/Keras/TensorFlow Lite)
As mentioned in the previous section, Keras works as an interface for TensorFlow 2, and they save the model in the same format as the frozen graph, which ends with `.pb`. Therefore, the TensorFlow/Keras/TensorflowLite operation is supported if it can be described by the TensorFlow raw operations listed below when the model is saved in the format of the frozen graph.

@<tbl:media/supported_tf.xlsx;Sheet1;TensorFlow Supported Operations>

# API Reference
## Function: mxq_compile
Compile a given model directly without creating an instance of "Model_Dict".
@<tbl:media/mxq_compile.xlsx;Sheet1;mxq_compile>
 
### Tips for choosing quantization methods
"Percentile" and "MaxPercentile" quantization methods each take a hyperparameter called @<i>percentile@</i>. An increase in this value corresponds to a broader quantization interval. To elaborate further, a higher @<i>percentile@</i> results in reduced overflow, albeit at the expense of accuracy.

The "MaxPercentile" method determines the percentile value from data that has been filtered once. As a result, a lower @<i>percentile@</i> is needed for "MaxPercentile" compared to the "Percentile" method. For instance, for the "Percentile" method, we suggest using a value of 0.9999 to 0.999999. For the "MaxPercentile" method, we recommend @<i>percentile@</i> between 0.9 and 0.9999.

The "is_quant_ch" argument enables channel-wise quantization. When set to True, the quantization is performed on a per-channel basis. This method is particularly useful for models, in which activations vary significantly across channels. However, it may take a longer time to compile the model.

The "quant_output" argument is used to determine the quantization method for the output layer. When the original model's output is various across the channels, it is recommended to set "ch" to keep channel-wise quantization. Otherwise, set "layer" to quantize the output layer as a whole.

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
 
## Function: make_calib
From the given images and preprocessing configuration, create the preprocessed numpy files and a txt file containing their paths.
@<tbl:media/make_calib.xlsx;Sheet1;make_calib>
 
## Fuction: make_calib_man
From given images and manually written function that takes an image path as input, create the preprocessed numpy files and a txt file containing their paths.
@<tbl:media/make_calib_man.xlsx;Sheet1;make_calib_man>
 
Example codes for using these functions are provided in the @<link:## Preparing Calibration Data> section.

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
 
@<tbl:media/pre_process.Pad.xlsx;Sheet1;Pad>
 
@<tbl:media/pre_process.Normalize.xlsx;Sheet1;Normalize>
 
@<tbl:media/pre_process.ResizeTorch.xlsx;Sheet1;ResizeTorch>
 
@<tbl:media/pre_process.Resize.xlsx;Sheet1;Resize>
 
@<tbl:media/pre_process.CenterCrop.xlsx;Sheet1;CenterCrop>
 
@<tbl:media/pre_process.SetOrder.xlsx;Sheet1;SetOrder>
 

# Open Source License Notice

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