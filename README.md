# CLIP_photo_search
OpenAI CLIP model for photo search based on OpenVINO acceleration

In this project, we use OpenVINO for OpenVINO CLip model acceleration, and construct one photo search demo with converted model.
![architecutre](https://github.com/juan-OY/CLIP_photo_search/assets/43125192/70e7cd4c-b8c7-4fc0-a80a-481ffa07da44)
### Input:
- A description text, encoded once
- The entire collection of images, encoded only once and will be indexed and resued later
### Output: 
- The K most similar images

## Step 1: Environment Setup 
Follow Below guide to set up the virtual environment
```
conda create -n openvino_env python=3.8
conda activate openvino_env
# Upgrade pip to the latest version to ensure compatibility with all dependencies
python -m pip install --upgrade pip==21.3.*
pip install -r requirements.txt
```
## Step 2: Model Conversion 
You can follow below guide to get converted CLIP image model and text model, and put them under ir folder, or even convert it to lower precision with nncf 
[OpenVINO notebook for clip model conversion](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/232-clip-language-saliency-map)

## Step 3: Image folder
In current code the images folder under data will be searched recursively, pls put your images under data folder

## Step 4: Run the code
```
python clip-search.py
```
Below please find a video show the demo

https://github.com/juan-OY/CLIP_photo_search/assets/43125192/a55dbf27-9766-4974-890c-214b891446c1

