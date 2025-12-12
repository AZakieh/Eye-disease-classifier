import unittest
from eye_classifier import EyeLandmarksDataset
from eye_classifier import EyeResNet
from eye_classifier import predict_image
import torch
import PIL
import os

class TestManager(unittest.TestCase):
    
    def setUp(self):
        self.EyeLandmark = EyeLandmarksDataset()
        self.EyeRes = EyeResNet()
    
    def test_EyeLandmarkDataset(self):
        self.image, self.label = self.EyeLandmark.__getitem__(0)
        
        self.assertEqual(self.image.shape, (3, 224, 224))
        self.assertIn(self.label, [0, 1])

    def test_EyeResnet(self):
        self.output = self.EyeRes(torch.randn(1, 3, 224, 224))
        self.assertEqual(self.output.shape, (1,2))

    def test_predict_image(self):
        self.test_image = PIL.Image.new(mode="RGB", size=(200,200))
        self.test_image.save("test_image.png")
        self.output = predict_image(self.EyeRes, "test_image.png")
        self.assertIn(self.output, [0, 1])

        os.remove("test_image.png")
        



    
