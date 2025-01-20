# Introduction
This repository contains source code for [	
RAIN: Real-time Animation Of Infinite Video Stream](https://arxiv.org/abs/2412.19489). A real-time implementation for video generataion on customer-level devices.

Project Page is [here](https://pscgylotti.github.io/pages/RAIN).

# Update Plan
- [x] Release Demo code
- [ ] Release inference pipeline code
- [ ] Release source code for training
- [ ] Update for a more interactive implementation

# Usage
## Installation
We recommend python `>=3.10`. Install pytorch on the [official website](https://pytorch.org/) firstly (We recommend torch `>=2.3.0`).
```
git clone https://github.com/Pscgylotti/RAIN.git
# clone repository from github
cd RAIN
pip install -r requirements_inference.txt  
# install requirements for inferencing
```

## Weights
You can download original RAIN weights from [Google Drive](https://drive.google.com/drive/folders/1xm8kdjWKgc1xbou63U7OhgSzqxzNUCsQ?usp=drive_link), [Huggingeface Hub](https://huggingface.co/Pscgylotti/RAIN-v0.1/tree/main), and then put them into `weights/torch/`.

You can get `'taesdv.pth'` from https://github.com/madebyollin/taesdv, and put it into `weights/torch/`.

Clone https://huggingface.co/stabilityai/sd-vae-ft-mse into `weights/`.

Download `'image_encoder'` folder and its contents from https://huggingface.co/lambdalabs/sd-image-variations-diffusers and put it into `weights/`.

Download `'dw-mm_ucoco.onnx'` from https://huggingface.co/hr16/UnJIT-DWPose/tree/main and `'yolox_s.onnx'` from https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0, and put them into `weights/onnx`. (You can choose to use `'dw-ll_ucoco-384.onnx'` and `'yolox_l.onnx'` from https://huggingface.co/yzd-v/DWPose/tree/main for higher accuracy).

(You can always redirect the weights directory in `configs/rain_morpher.yaml`)

## TensorRT 
In `configs/rain_morpher.yaml`, modify `tensorrt: False` into `tensorrt: True` to enable TensorRT acceleration. In the first launch it will take about ten minutes to compile the model.

## Demo Launch
Simply execute `python gradio_app.py` and open `http://localhost:7860/` in browser (Usually the port is `7860`).

Upload an upper-half-body potrait of any anime characters, fuse reference and turn on the web camera. Click on start to launch face morphing. You may need to adjust some morphing parameters to fit your face with the character face (Especially for the eye-related parameters. The eyes generally fail to synthesize with incompatible parameters). 

## Hardware Requirement
It generally takes about `12 GiB` of Device RAM to run the whole inference demo. However, you can unload the reference part after fusing the reference image. Then the synthesis-only model requires about `8 GiB` of Device RAM to run.

# Acknowledgment

Our work is based on [AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master), [Moore-AnimateAnyone](https://github.com/MooreThreads/Moore-AnimateAnyone), [Open-AnimateAnyone](https://github.com/guoqincode/Open-AnimateAnyone), [AnimateDiff](https://github.com/guoyww/AnimateDiff), [DWPose](https://github.com/IDEA-Research/DWPose), [TinyAutoencoder](https://github.com/madebyollin/taesdv), [AnimeFaceDetector](https://github.com/hysts/anime-face-detector). Thanks to these teams/authors for their work.

Special thanks to [CivitAI Community](https://civit.ai) and [YODOYA](https://www.pixiv.net/users/101922785) for example images. Thanks to [Jianwen Meng](mailto:jwmeng@mail.ustc.edu.cn) for pipeline design.