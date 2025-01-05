import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

class VQ4DiT:
    def __init__(
        self,
        model: nn.Module,
        bit_width: int = 2,
        sub_vector_dim: int = 4,
        candidate_size: int = 2,
        batch_size: int = 16,
        num_iterations: int = 500,
        learning_rate_ratios: float = 5e-2,
        learning_rate_others: float = 1e-4,
    ):
        """Initialize VQ4DiT quantization.
        
        Args:
            model: DiT model to quantize
            bit_width: Target quantization bit width (2 or 3)
            sub_vector_dim: Dimension of weight sub-vectors
            candidate_size: Number of candidate assignments per sub-vector
            batch_size: Batch size for calibration
            num_iterations: Number of calibration iterations
            learning_rate_ratios: Learning rate for assignment ratios
            learning_rate_others: Learning rate for other parameters
        """
        self.model = model
        self.num_centroids = 2 ** bit_width
        self.sub_vector_dim = sub_vector_dim
        self.candidate_size = candidate_size
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.lr_ratios = learning_rate_ratios
        self.lr_others = learning_rate_others
        
        # Store quantization state
        self.codebooks: Dict[str, torch.Tensor] = {}
        self.assignments: Dict[str, torch.Tensor] = {}
        self.candidate_assignments: Dict[str, torch.Tensor] = {}
        self.assignment_ratios: Dict[str, torch.Tensor] = {}

    def _init_layer_quantization(self, name: str, weight: torch.Tensor) -> None:
        """Initialize quantization parameters for a layer."""
        # Reshape weight into sub-vectors
        orig_shape = weight.shape
        num_sub_vectors = np.prod(orig_shape) // self.sub_vector_dim
        weight_reshaped = weight.reshape(num_sub_vectors, self.sub_vector_dim)

        # Initialize codebook using K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.num_centroids)
        kmeans.fit(weight_reshaped.cpu().numpy())
        
        self.codebooks[name] = torch.from_numpy(kmeans.cluster_centers_).to(weight.device)
        
        # Find candidate assignments based on Euclidean distance
        distances = torch.cdist(weight_reshaped, self.codebooks[name])
        _, candidates = torch.topk(distances, k=self.candidate_size, largest=False, dim=1)
        
        self.candidate_assignments[name] = candidates
        self.assignment_ratios[name] = torch.ones_like(candidates, dtype=torch.float32) / self.candidate_size

    def _reconstruct_weight(self, name: str) -> torch.Tensor:
        """Reconstruct weight using weighted average of candidate assignments."""
        codebook = self.codebooks[name]
        candidates = self.candidate_assignments[name]
        ratios = self.assignment_ratios[name]
        
        # Weighted average reconstruction
        reconstructed = torch.zeros_like(candidates, dtype=torch.float32)
        for i in range(self.candidate_size):
            reconstructed += ratios[:, i:i+1] * codebook[candidates[:, i]]
            
        return reconstructed

    def _zero_data_calibration(self) -> None:
        """Perform zero-data block-wise calibration."""
        optimizer = torch.optim.RMSprop([
            {'params': list(self.codebooks.values()), 'lr': self.lr_others},
            {'params': list(self.assignment_ratios.values()), 'lr': self.lr_ratios}
        ])

        for iter in tqdm(range(self.num_iterations)):
            # Generate random noise input
            noise = torch.randn(self.batch_size, *self.model.input_shape)
            noise = noise.to(next(self.model.parameters()).device)
            
            # Forward pass with original weights
            with torch.no_grad():
                orig_outputs = []
                for block in self.model.blocks:
                    noise = block(noise)
                    orig_outputs.append(noise.clone())

            # Forward pass with quantized weights
            optimizer.zero_grad()
            quant_outputs = []
            
            for block_idx, block in enumerate(self.model.blocks):
                # Temporarily replace weights with quantized versions
                stored_weights = {}
                for name, param in block.named_parameters():
                    if name in self.codebooks:
                        stored_weights[name] = param.data.clone()
                        param.data = self._reconstruct_weight(name).reshape(param.shape)
                
                noise = block(noise)
                quant_outputs.append(noise.clone())
                
                # Restore original weights
                for name, param in block.named_parameters():
                    if name in stored_weights:
                        param.data = stored_weights[name]

            # Compute block-wise MSE loss
            mse_loss = sum(torch.mean((orig - quant) ** 2) 
                          for orig, quant in zip(orig_outputs, quant_outputs))
            
            # Add ratio regularization loss
            ratio_loss = sum(torch.mean((1 - 2 * ratios - 1) / ratios.numel())
                           for ratios in self.assignment_ratios.values())
            
            loss = mse_loss + ratio_loss
            loss.backward()
            optimizer.step()

            # Update assignments if ratios have converged
            if iter > 0 and ratio_loss < 1e-4:
                self._update_final_assignments()
                break

    def _update_final_assignments(self) -> None:
        """Update final assignments based on highest ratios."""
        for name in self.candidate_assignments:
            ratios = self.assignment_ratios[name]
            best_candidates = torch.argmax(ratios, dim=1)
            self.assignments[name] = torch.gather(
                self.candidate_assignments[name], 1, 
                best_candidates.unsqueeze(1)
            ).squeeze(1)

    def quantize(self) -> nn.Module:
        """Quantize the model using VQ4DiT."""
        # Initialize quantization for each layer
        for name, param in self.model.named_parameters():
            if len(param.shape) > 1:  # Only quantize weight matrices
                self._init_layer_quantization(name, param.data)
        
        # Perform calibration
        self._zero_data_calibration()
        
        # Apply final quantized weights
        for name, param in self.model.named_parameters():
            if name in self.assignments:
                param.data = self._reconstruct_weight(name).reshape(param.shape)
        
        return self.model

# Usage example:
def quantize_dit_model(model, bit_width=2):
    quantizer = VQ4DiT(model, bit_width=bit_width)
    quantized_model = quantizer.quantize()
    return quantized_model
