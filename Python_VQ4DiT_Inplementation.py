import torch
import numpy as np
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

# Load the Diffusion Transformer model (DiT)
model_name = "CompVis/stable-diffusion-v1-4"  # Example model, replace with your target DiT model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def initialize_codebook_and_assignments(weight, num_clusters=256):
    """
    Initialize the codebook and assignments using K-Means clustering.
    Args:
        weight (np.array): Weight matrix from the model.
        num_clusters (int): Number of clusters for the codebook.
    Returns:
        codebook (np.array): Cluster centroids.
        assignments (np.array): Indices of codebook vectors for each weight sub-vector.
    """
    reshaped_weight = weight.reshape(-1, 1)  # Flatten weights for clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(reshaped_weight)
    codebook = kmeans.cluster_centers_.flatten()
    assignments = kmeans.labels_.reshape(weight.shape)
    return codebook, assignments

def quantize_weights(weight, codebook, assignments):
    """
    Quantize the weights using the codebook and assignments.
    Args:
        weight (np.array): Original weight matrix.
        codebook (np.array): Cluster centroids (codebook).
        assignments (np.array): Indices of codebook vectors for each weight sub-vector.
    Returns:
        quantized_weight (np.array): Quantized weight matrix.
    """
    return codebook[assignments]

def vq4dit_quantize(model, num_clusters=256):
    """
    Apply vector quantization (VQ4DiT) to the weights of a transformer model.
    Args:
        model (torch.nn.Module): Transformer model to quantize.
        num_clusters (int): Number of clusters for the codebook.
    Returns:
        quantized_model (torch.nn.Module): Model with quantized weights.
    """
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:  # Only quantize weight matrices
            weight = param.data.cpu().numpy()
            
            # Initialize codebook and assignments
            codebook, assignments = initialize_codebook_and_assignments(weight, num_clusters=num_clusters)
            
            # Quantize weights
            quantized_weight = quantize_weights(weight, codebook, assignments)
            
            # Replace original weights with quantized weights
            param.data = torch.from_numpy(quantized_weight).to(param.device)
    
    return model

# Quantize the model
quantized_model = vq4dit_quantize(model, num_clusters=256)

# Save the quantized model locally
quantized_model.save_pretrained("./quantized_model")
tokenizer.save_pretrained("./quantized_model")

# Optionally push the model to Hugging Face Hub
from huggingface_hub import HfApi, HfFolder

api = HfApi()
username = api.whoami()["name"]
repo_id = f"{username}/quantized-{model_name.split('/')[-1]}"
api.create_repo(repo_id, private=False)

quantized_model.push_to_hub(repo_id, use_auth_token=HfFolder.get_token())
tokenizer.push_to_hub(repo_id, use_auth_token=HfFolder.get_token())

print("Quantized model successfully pushed to Hugging Face Hub.")
