"""
Visual Embedding Generator and Similarity Analysis for VisionMate Personal Objects.

This module provides feature vector extraction and cosine similarity matching for cropped
object images using a pretrained PyTorch ResNet-50 deep neural network backbone.

Domain-Specific Concepts & Architectural Decisions:
---------------------------------------------------
1. What an Embedding Represents:
   An object embedding is a dense, high-dimensional numerical vector (2048-D float32)
   capturing rich semantic visual attributes, spatial features, textures, color distributions,
   and structural patterns of a cropped target object image region.

2. Why ImageNet Classification Layer Output is Removed:
   The original ResNet-50 network terminates with a fully-connected layer (`fc`) projecting
   features to 1000 ImageNet class probability logits. By replacing `model.fc` with `torch.nn.Identity()`,
   we extract feature representations directly from the pen-ultimate global average pooling layer.
   This provides a domain-agnostic 2048-dimensional feature vector suitable for open-set visual search
   and fine-grained instance comparison across arbitrary personal objects rather than restricting to ImageNet classes.

3. Why Embeddings are L2-Normalized:
   L2-normalization maps every feature vector onto a unit hypersphere (||v||_2 = 1.0).
   Normalizing eliminates vector magnitude variations caused by lighting changes, exposure differences,
   or input image dimensions. This ensures downstream visual comparison depends purely on feature direction
   rather than illumination intensity or scale.

4. How Cosine Similarity is Interpreted:
   Cosine similarity measures the angular cosine between two unit feature vectors in 2048-dimensional vector space:
   - 1.0: Identical visual feature direction (maximum visual similarity).
   - 0.0: Orthogonal visual feature directions (unrelated / independent visual features).
   - -1.0: Diametrically opposite feature vectors.
   Because all generated embeddings are L2-normalized upon extraction, cosine similarity simplifies to the matrix/vector dot product.
"""

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from typing import Union, List, Optional, Any


class PersonalObjectEmbedder:
    """
    Extracts 2048-dimensional normalized visual feature embeddings from personal object image crops
    using a pretrained ResNet-50 vision backbone.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        device: Optional[str] = None
    ):
        """
        Initialize the PersonalObjectEmbedder with a ResNet-50 model and preprocessing pipeline.

        Args:
            model_name (str): Name of vision backbone. Defaults to "resnet50".
            device (str, optional): Target PyTorch device ("cuda:0", "cpu", etc.).
                                   Automatically defaults to "cuda:0" if CUDA is available, else "cpu".
        """
        if model_name.lower() != "resnet50":
            raise ValueError(f"Unsupported model name '{model_name}'. Only 'resnet50' is currently supported.")

        # Determine target execution device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load ResNet-50 with default weights and preprocessing transforms
        self.weights = torchvision.models.ResNet50_Weights.DEFAULT
        self.model = torchvision.models.resnet50(weights=self.weights)

        # Replace final classification head with Identity to output 2048-D global feature vector
        self.model.fc = torch.nn.Identity()

        # Set evaluation mode and move to selected device
        self.model.eval()
        self.model.to(self.device)

        # Preprocessing transforms associated with ResNet50_Weights.DEFAULT
        self.transforms = self.weights.transforms()

    def _prepare_pil_image(self, image: Any, is_bgr: bool = True) -> Image.Image:
        """
        Validate and convert an input image (OpenCV BGR, RGB NumPy array, or PIL Image)
        into a valid RGB PIL Image instance.

        Args:
            image: Input image (PIL.Image.Image or numpy.ndarray).
            is_bgr (bool): Whether 3-channel NumPy arrays are in OpenCV BGR format. Default True.

        Returns:
            PIL.Image.Image: Converted RGB PIL Image.
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        if isinstance(image, Image.Image):
            w, h = image.size
            if w <= 0 or h <= 0:
                raise ValueError(f"Invalid PIL Image dimensions: width={w}, height={h}.")
            return image.convert("RGB")

        if isinstance(image, np.ndarray):
            if image.size == 0 or image.ndim < 2:
                raise ValueError("Empty or invalid NumPy image array provided.")

            h, w = image.shape[:2]
            if h <= 0 or w <= 0:
                raise ValueError(f"Invalid NumPy image dimensions: height={h}, width={w}.")

            if image.ndim == 2:  # Grayscale
                rgb_arr = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.ndim == 3:
                channels = image.shape[2]
                if channels == 3:
                    if is_bgr:
                        rgb_arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_arr = image.copy()
                elif channels == 4:
                    if is_bgr:
                        rgb_arr = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                    else:
                        rgb_arr = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                else:
                    raise ValueError(f"Unsupported number of image channels: {channels}.")
            else:
                raise ValueError(f"Unsupported NumPy array shape: {image.shape}.")

            return Image.fromarray(rgb_arr)

        raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL Image or numpy.ndarray.")

    def embed_image(self, image: Any, is_bgr: bool = True) -> np.ndarray:
        """
        Generate a 2048-dimensional L2-normalized float32 feature embedding for a single image.

        Args:
            image: Input image (OpenCV BGR array, RGB NumPy array, or PIL Image).
            is_bgr (bool): Set to True if NumPy array is OpenCV BGR format (default True).

        Returns:
            numpy.ndarray: L2-normalized feature vector of shape (2048,) and dtype float32.
        """
        pil_img = self._prepare_pil_image(image, is_bgr=is_bgr)

        # Apply official torchvision ResNet50 preprocessing
        tensor_img = self.transforms(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features_tensor = self.model(tensor_img)

        features = features_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        # Safe L2-normalization
        norm = float(np.linalg.norm(features))
        if norm > 1e-12:
            features = features / norm
        else:
            features = np.zeros_like(features, dtype=np.float32)

        return features

    def batch_embed(self, images: List[Any], is_bgr: bool = True) -> np.ndarray:
        """
        Generate L2-normalized 2048-dimensional feature embeddings for a batch of images.

        Args:
            images (list): List of input images (OpenCV BGR, RGB NumPy, or PIL Images).
            is_bgr (bool): Set to True if NumPy arrays are OpenCV BGR format (default True).

        Returns:
            numpy.ndarray: Matrix of shape (N, 2048) containing L2-normalized float32 embeddings.
        """
        if images is None:
            raise ValueError("Input images list cannot be None.")

        if not isinstance(images, (list, tuple)):
            raise TypeError(f"Expected list or tuple of images, got {type(images)}.")

        if len(images) == 0:
            return np.empty((0, 2048), dtype=np.float32)

        preprocessed_tensors = []
        for idx, img in enumerate(images):
            try:
                pil_img = self._prepare_pil_image(img, is_bgr=is_bgr)
                tensor_img = self.transforms(pil_img)
                preprocessed_tensors.append(tensor_img)
            except Exception as e:
                raise ValueError(f"Error processing image at index {idx}: {e}") from e

        batch_tensor = torch.stack(preprocessed_tensors).to(self.device)

        with torch.no_grad():
            features_tensor = self.model(batch_tensor)

        features = features_tensor.cpu().numpy().astype(np.float32)

        # Safe row-wise L2-normalization
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms > 1e-12, norms, 1.0)
        normalized_features = features / norms

        return normalized_features

    @staticmethod
    def similarity(embedding_a: Any, embedding_b: Any) -> float:
        """
        Calculate cosine similarity between two 2048-dimensional feature vectors.

        Args:
            embedding_a: First feature vector (numpy.ndarray or list).
            embedding_b: Second feature vector (numpy.ndarray or list).

        Returns:
            float: Cosine similarity score bounded in range [-1.0, 1.0].
        """
        if embedding_a is None or embedding_b is None:
            raise ValueError("Input embeddings cannot be None.")

        arr_a = np.asarray(embedding_a, dtype=np.float32).ravel()
        arr_b = np.asarray(embedding_b, dtype=np.float32).ravel()

        if arr_a.size != 2048 or arr_b.size != 2048:
            raise ValueError(
                f"Invalid embedding dimensions. Expected length 2048 for both vectors, "
                f"but got {arr_a.size} and {arr_b.size}."
            )

        if np.isnan(arr_a).any() or np.isnan(arr_b).any():
            raise ValueError("Input embeddings contain NaN values.")

        norm_a = float(np.linalg.norm(arr_a))
        norm_b = float(np.linalg.norm(arr_b))

        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0

        dot_product = float(np.dot(arr_a, arr_b))
        sim = dot_product / (norm_a * norm_b)

        return float(np.clip(sim, -1.0, 1.0))
