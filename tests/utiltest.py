import unittest
import torch
from torch import nn

import sys

import kgbench as kg
from kgbench import tic, toc

import torch
import numpy as np

class TestUtil(unittest.TestCase):

    def test_batching(self):

        amplus = kg.load('amplus')

        batched = kg.to_tensorbatches(amplus.get_pil_images(), use_torch=True)

        for btch in batched:
            btch = np.array(batched[17]).astype(np.float)/255.

