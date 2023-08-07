#!/usr/bin/env python
# coding: utf-8

# ## Convert to OpenVINO™ Intermediate Representation (IR) Format
# 
# The process of building a saliency map can be quite time-consuming. To speed it up, you will use OpenVINO. OpenVINO is an inference framework designed to run pre-trained neural networks efficiently. One way to use it is to convert a model from its original framework representation to an OpenVINO Intermediate Representation (IR) format and then load it for inference. The model currently uses PyTorch. To get an IR, you need to first convert the PyTorch model to the ONNX format. It can be done with the `torch.onnx.export` function. See the [PyTorch documentation](https://pytorch.org/docs/stable/onnx.html) for more information on ONNX conversion.

# Now, you have two separate models for text and images, stored on disk and ready to be loaded and inferred with OpenVINO™.
# 
# ## Inference with OpenVINO™
# 
# 1. Create an instance of the `Core` object that will handle any interaction with OpenVINO runtime for you.
# 1. Use the `core.read_model` method to load the model into memory.
# 1. Compile the model with the `core.compile_model` method for a particular device to apply device-specific optimizations.
# 1. Use the compiled model for inference.

from openvino.runtime import Core
import time
import os
from PIL import Image
import numpy as np
from tqdm import tqdm  # Import tqdm for progress monitoring
from transformers import CLIPProcessor
import matplotlib.pyplot as plt
import gradio as gr
import io
import sys

core = Core()
text_model_path = "ir/clip-vit-base-patch16_text.xml"
image_model_path = "ir/clip-vit-base-patch16_image.xml"

text_model = core.read_model(text_model_path)
image_model = core.read_model(image_model_path)


device = "CPU"

text_model = core.compile_model(model=text_model, device_name=device)
image_model = core.compile_model(model=image_model, device_name=device)

#cache_directory = "~/.cache/huggingface/hub"  # Set the path to your cache directory here

# load preprocessor for model input
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
data_folder = "data"

#query = "a cute turtle"
# Initialize lists to store image embeddings and image file paths
image_embeds_list = []
image_file_paths = []

    
# Populate image_file_paths list with the file paths of the images in the data folder
for root, dirs, files in os.walk(data_folder):
    for file in files:
        if file.endswith((".png", ".jpg", ".jpeg" ,".JPEG")):
            image_file_paths.append(os.path.join(root, file))
image_file_paths_len = len(image_file_paths)            
print(f"image file path length: {image_file_paths_len}")


def calculate_and_cache_image_embeddings(image_file_paths, model, existing_data):
    new_data = []
    
    for image_path in tqdm(image_file_paths, desc="Processing images"):
        if image_path not in [item[0] for item in existing_data]:
            with Image.open(image_path) as image:
                inputs = processor(images=image, return_tensors="np")
                start_time = time.time()
                image_inputs = inputs.pop("pixel_values")
                image_embeds = model(image_inputs)[model.output()]
                end_time = time.time()
                time_pass = end_time - start_time
                print(f"Time to compute image embeddings for {image_path}: {time_pass} seconds")
                new_data.append((image_path, image_embeds))
    if new_data:
        updated_data = np.concatenate((existing_data, new_data), axis=0)
        np.save("image_embeddings.npy", updated_data)
    else:
        updated_data = existing_data

    return updated_data

def remove_unused_entries(existing_data, current_image_file_paths):
    updated_data = [(path, embedding) for path, embedding in existing_data if path in current_image_file_paths]
    np.save("image_embeddings.npy", updated_data)
    return updated_data
    
def calculate_text_embeddings(query, model):
    text_inputs = processor(text=[query], return_tensors="np")
    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs["attention_mask"]
    input_data = {'input_ids': input_ids, 'attention_mask': attention_mask}
    text_embeds = text_model(inputs=input_data)[text_model.output()]
    return text_embeds

# Compute and cache image embeddings
if os.path.exists("image_embeddings.npy"):
    start_time = time.time()
    existing_data = np.load("image_embeddings.npy", allow_pickle=True)  
    end_time = time.time()
    time_pass = end_time - start_time
    print("When to enter the path exists path")
    print(f"image_embeddings.npy, size: {existing_data.shape[0]}")
else:
    existing_data = np.empty((0, 2), dtype=[('path', '<U255'), ('embedding', 'O')])
    print("no existing image embeddings found")

all_images_embeds = calculate_and_cache_image_embeddings(image_file_paths, image_model, existing_data)
print("Embeddings calculation and caching completed.")

# Update all_image_embeds with new data
all_image_embeds = np.concatenate([item[1] for item in all_images_embeds], axis=0)
print("all_image_embeds updated.")
# Remove unused entries
updated_existing_data = remove_unused_entries(existing_data, image_file_paths)
print("Unused entries removed.")
print("addddddddddddddddddddddddd debug to return")
    
def cosine_similarity(one, other):
    return one @ other.T / (np.linalg.norm(one) * np.linalg.norm(other))

def spherical_distance(text_embeds, all_image_embeds):
    text_embeds_norm = np.apply_along_axis(normalize_vector, 1, text_embeds)
    image_embeds_norm = np.apply_along_axis(normalize_vector, 1, all_image_embeds)
    dot_products = np.dot(text_embeds_norm, image_embeds_norm.T)
    dot_products = np.clip(dot_products, -1, 1)
    spherical_dists = np.arccos(dot_products)
    return spherical_dists

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
    
## photo search with 2 methods
def find_similar_photos(image_file_paths, text_embeds, all_image_embeds, similarity_measure="cosine", top_k=5):
    if similarity_measure == "cosine":
        distances = cosine_similarity(text_embeds, all_image_embeds)
        # Normalize cosine distances to be within [0, 1]
        similarities = 0.5 + 0.5 * distances
    elif similarity_measure == "spherical":
        distances = spherical_distance(text_embeds, all_image_embeds)
        # Inverse the spherical distances to convert to similarities
        max_dist = np.pi
        similarities = 1.0 - (distances / max_dist)
    else:
        raise ValueError("Invalid similarity_measure. Use 'cosine' or 'spherical'.")

    # Combine images with their similarity scores
    image_score_pairs = list(zip(image_file_paths, similarities[0]))

    # Sort the pairs based on the similarity scores in descending order
    image_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # Get the top-k similar image paths and their scores
    top_similar_image_paths = [pair[0] for pair in image_score_pairs[:top_k]]
    top_similar_scores = [pair[1] for pair in image_score_pairs[:top_k]]

    return top_similar_image_paths, top_similar_scores


def get_similar_photos(query, data_folder):
    print("get_similar_photos in")
    start_time1 = time.time()
    # Process the text query
    text_embeds = calculate_text_embeddings(query, text_model)

    start_time = time.time()
    # Find similar photos using cosine similarity
    cosine_similar_image_paths, cosine_similar_scores = find_similar_photos(
        image_file_paths, text_embeds, all_image_embeds, similarity_measure="cosine", top_k=5
    )
    end_time = time.time()
    time_pass = end_time - start_time
    print(f"111 time to caculate similarity image path: {time_pass}")
    # Return the top 1 similar photo and the top 2-5 similar photos as NumPy arrays
    top_similar_photo_path = cosine_similar_image_paths[0]
    top_similar_photos_paths_2_to_5 = cosine_similar_image_paths[1:]
       
    #print(top_similar_photo_path)
    #print(top_similar_photos_paths_2_to_5)
    print("get_similar_photos")
    end_time1 = time.time()
    time_pass = end_time1 - start_time
    print(f"222 time to get_similar_photos: {time_pass}")
    return top_similar_photo_path, top_similar_photos_paths_2_to_5


# Find similar photos using spherical distance
#spherical_similar_image_paths, spherical_similar_scores = find_similar_photos(image_file_paths, text_embeds, all_image_embeds,
#                                                                            similarity_measure="spherical", top_k=5)


# Define the Gradio interface
text_query_input = gr.inputs.Textbox(lines=2, label="Enter the text query:")
data_folder_input = gr.inputs.Textbox(default="data", label="Enter the path to the data folder containing images:")
#output_image = gr.outputs.Image(type="numpy", label="Top Similar Photo")
#output_images = gr.outputs.Image(type="numpy", label="Top 2-5 Similar Photos")

# Set live=True to remove the submit button for real-time updates
interface = gr.Interface(
    fn=get_similar_photos,
    inputs=[text_query_input, data_folder_input],
    outputs=[
        gr.Image(type="filepath", label="Top Similar Photo", width=224, height=224),
        gr.Gallery(label="Other Similar Photo").style(columns=4, display="grid", grid_template_columns="repeat(4, 60px)")
    ],
    live=False,  #with a submit button
    title="Similar Photo Search",
    description="Find similar photos based on a text query using CLIP embeddings.",
)
# Apply style to the Gallery component separately
#interface.outputs[1].style(columns=4)
# Launch the Gradio interface
interface.launch()

print("interface.launch() after" )



