This script is a (currently draft in development) of an implementation for the VQ4DiT technique for quantization outlined in the paper "" by Deng et al.

Paper:
https://huggingface.co/papers/2408.17131
Aug 30th 2024
Authors:
Juncan Deng
,
Shuaiting Li
,
Zeyu Wang
,
Hong Gu
,
Kedong Xu
,
Kejie Huang
Abstract
The Diffusion Transformers Models (DiTs) have transitioned the network architecture from traditional UNets to transformers, demonstrating exceptional capabilities in image generation. Although DiTs have been widely applied to high-definition video generation tasks, their large parameter size hinders inference on edge devices. Vector quantization (VQ) can decompose model weight into a codebook and assignments, allowing extreme weight quantization and significantly reducing memory usage. In this paper, we propose VQ4DiT, a fast post-training vector quantization method for DiTs. We found that traditional VQ methods calibrate only the codebook without calibrating the assignments. This leads to weight sub-vectors being incorrectly assigned to the same assignment, providing inconsistent gradients to the codebook and resulting in a suboptimal result. To address this challenge, VQ4DiT calculates the candidate assignment set for each weight sub-vector based on Euclidean distance and reconstructs the sub-vector based on the weighted average. Then, using the zero-data and block-wise calibration method, the optimal assignment from the set is efficiently selected while calibrating the codebook. VQ4DiT quantizes a DiT XL/2 model on a single NVIDIA A100 GPU within 20 minutes to 5 hours depending on the different quantization settings. Experiments show that VQ4DiT establishes a new state-of-the-art in model size and performance trade-offs, quantizing weights to 2-bit precision while retaining acceptable image generation quality.

*The author of this repo and script is not an author of the original paper and have no guarentees on correctness nor functionality.*

**This program does the following: **

1. Pull a model from the Hugging Face Hub:

Only Diffusion Transformer Models (DiTs) are applicable as per the VQ4DiT method. Examples include image generation models or video generation models that use the transformer architecture instead of UNet.

2. Apply the VQ4DiT method:

VQ4DiT is a post-training vector quantization method specifically designed for DiTs. It involves:
Decomposing model weights into a codebook and assignments using K-Means clustering.
Calculating candidate assignment sets based on Euclidean distance.
Calibrating the codebook and assignments using a zero-data and block-wise calibration strategy.

3. Save and Push the Quantized Model to the Hugging Face Hub:

Save the quantized model locally and then use the Hugging Face Hub API to push it.

