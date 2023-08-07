# CLIP_photo_search
OpenAI CLIP model for photo search based on OpenVINO acceleration

In this project, we use OpenVINO for OpenVINO CLip model acceleration, and construct one photo search demo with converted model.

## Step 1: Environment Setup 
Follow OpenVINO guide to set up the virtual environment as in below guide 
[OpenVINO Environment Setup](https://github.com/openvinotoolkit/openvino_notebooks/wiki/Ubuntu)
```
conda create -n openvino_env python=3.8
conda activate openvino_env
# Upgrade pip to the latest version to ensure compatibility with all dependencies
python -m pip install --upgrade pip==21.3.*
pip install -r requirements.txt
```
## Step 2: Model Conversion 
please follow below guide to get converted CLIP image model and text model 

Reference: 
[OpenVINO notebook for clip model conversion](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/232-clip-language-saliency-map)

