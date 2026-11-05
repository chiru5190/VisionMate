"""
Unit tests for personal/embeddings.py (PersonalObjectEmbedder).
"""

import sys
import os
import unittest
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

from personal.embeddings import PersonalObjectEmbedder


class TestPersonalObjectEmbedder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Initializing PersonalObjectEmbedder for unit tests...")
        cls.embedder = PersonalObjectEmbedder()

    def test_1_initialization_and_device(self):
        """Test default initialization and device selection."""
        self.assertIsNotNone(self.embedder.model)
        self.assertIsNotNone(self.embedder.transforms)
        print(f"Active embedder device: {self.embedder.device}")

        # Explicit CPU initialization
        cpu_embedder = PersonalObjectEmbedder(device="cpu")
        self.assertEqual(str(cpu_embedder.device), "cpu")

        # Unsupported model name reject
        with self.assertRaises(ValueError):
            PersonalObjectEmbedder(model_name="vgg16")

    def test_2_embed_single_image_formats(self):
        """Test embedding extraction for OpenCV BGR, RGB numpy, and PIL image inputs."""
        # 1. OpenCV BGR numpy array
        bgr_array = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        emb_bgr = self.embedder.embed_image(bgr_array, is_bgr=True)
        self.assertEqual(emb_bgr.shape, (2048,))
        self.assertEqual(emb_bgr.dtype, np.float32)
        norm_bgr = np.linalg.norm(emb_bgr)
        self.assertAlmostEqual(norm_bgr, 1.0, places=5)

        # 2. RGB numpy array
        rgb_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        emb_rgb = self.embedder.embed_image(rgb_array, is_bgr=False)
        self.assertEqual(emb_rgb.shape, (2048,))
        norm_rgb = np.linalg.norm(emb_rgb)
        self.assertAlmostEqual(norm_rgb, 1.0, places=5)

        # 3. PIL Image
        pil_img = Image.fromarray(rgb_array)
        emb_pil = self.embedder.embed_image(pil_img)
        self.assertEqual(emb_pil.shape, (2048,))
        norm_pil = np.linalg.norm(emb_pil)
        self.assertAlmostEqual(norm_pil, 1.0, places=5)

        # Similarity between RGB array and its identical PIL conversion must be ~1.0
        sim_self = self.embedder.similarity(emb_rgb, emb_pil)
        self.assertAlmostEqual(sim_self, 1.0, places=5)

    def test_3_batch_embed(self):
        """Test batch embedding extraction."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (150, 120, 3), dtype=np.uint8)
        img3 = Image.fromarray(np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8))

        batch_emb = self.embedder.batch_embed([img1, img2, img3])
        self.assertEqual(batch_emb.shape, (3, 2048))
        self.assertEqual(batch_emb.dtype, np.float32)

        for i in range(3):
            norm_i = np.linalg.norm(batch_emb[i])
            self.assertAlmostEqual(norm_i, 1.0, places=5)

        # Test empty batch
        empty_batch = self.embedder.batch_embed([])
        self.assertEqual(empty_batch.shape, (0, 2048))

    def test_4_similarity_computation(self):
        """Test cosine similarity function."""
        # Identical vector
        vec_a = np.random.randn(2048).astype(np.float32)
        vec_a = vec_a / np.linalg.norm(vec_a)

        sim_identical = PersonalObjectEmbedder.similarity(vec_a, vec_a)
        self.assertAlmostEqual(sim_identical, 1.0, places=5)

        # Orthogonal vector test
        vec_b = np.random.randn(2048).astype(np.float32)
        vec_b = vec_b / np.linalg.norm(vec_b)
        sim_diff = PersonalObjectEmbedder.similarity(vec_a, vec_b)
        self.assertGreaterEqual(sim_diff, -1.0)
        self.assertLessEqual(sim_diff, 1.0)

        # Zero vector handling
        zero_vec = np.zeros(2048, dtype=np.float32)
        sim_zero = PersonalObjectEmbedder.similarity(vec_a, zero_vec)
        self.assertEqual(sim_zero, 0.0)

    def test_5_validation_and_errors(self):
        """Test input validation and error rejection."""
        # 1. Reject None image
        with self.assertRaises(ValueError):
            self.embedder.embed_image(None)

        # 2. Reject empty numpy array
        empty_array = np.array([], dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.embedder.embed_image(empty_array)

        # 3. Reject 0-dimension image shape
        zero_shape_array = np.zeros((0, 100, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.embedder.embed_image(zero_shape_array)

        # 4. Reject invalid similarity dimension
        invalid_dim_vec = np.random.randn(512).astype(np.float32)
        valid_vec = np.random.randn(2048).astype(np.float32)
        with self.assertRaises(ValueError):
            PersonalObjectEmbedder.similarity(invalid_dim_vec, valid_vec)

        # 5. Reject None in similarity
        with self.assertRaises(ValueError):
            PersonalObjectEmbedder.similarity(None, valid_vec)

        # 6. Reject None in batch_embed
        with self.assertRaises(ValueError):
            self.embedder.batch_embed(None)


if __name__ == '__main__':
    unittest.main()
