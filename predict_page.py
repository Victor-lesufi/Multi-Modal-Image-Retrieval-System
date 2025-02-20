import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# @st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    return model, processor

# Function to retrieve image embeddings
def get_image_embeddings(image_files, image_folder, processor, model):
    images = []
    for image_file in image_files:
        try:
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB format
            images.append(image)
        except Exception as e:
            st.warning(f"Unable to process image {image_file}. Error: {e}")
    image_inputs = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
    image_embeddings = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # Normalize
    return image_embeddings

# Function to get text embeddings
def get_text_embedding(query, processor, model):
    text_inputs = processor(text=query, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
    text_embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # Normalize
    return text_embedding

# Show prediction page function
def show_predict_page():
    st.title("Multi-Modal Image Retrieval System ")
    st.write("""### Enter a query here""")
    
    # Load model and processor
    model, processor = load_model()
    
    
    user_query = st.text_input("Enter your query:")
    
    if user_query:
        st.write(f"Fetching: '{user_query}'")
        
        
        image_folder = 'test_data'  
        
        # Check if the folder exists
        if not os.path.exists(image_folder):
            st.error(f"The folder '{image_folder}' does not exist. Please check the path.")
            return
        
        # List all image files in the folder
        image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not image_files:
            st.write("No images found in the specified folder.")
            return
        
        # Get image embeddings
        image_embeddings = get_image_embeddings(image_files, image_folder, processor, model)
        
        # Get text embedding for the query
        text_embedding = get_text_embedding(user_query, processor, model)
        
        # Compute cosine similarities
        similarities = cosine_similarity(text_embedding.cpu().numpy(), image_embeddings.cpu().numpy())
        
        # Sort images based on similarity score in descending order
        sorted_indices = np.argsort(similarities[0])[::-1]
        sorted_image_files = [image_files[i] for i in sorted_indices]
        sorted_similarities = similarities[0][sorted_indices]
        
        # Display all images sorted by similarity
        st.write("Top K images:")
        for i, (image_file, similarity) in enumerate(zip(sorted_image_files, sorted_similarities)):
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)
            st.image(image, caption=f"{i + 1}. {image_file} (Similarity: {similarity:.4f})", use_container_width=True)

# Run the app
if __name__ == "__main__":
    show_predict_page()