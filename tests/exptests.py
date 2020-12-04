import unittest
import torch
from torch import nn

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../experiments')))

import rgcn

class TestExps(unittest.TestCase):

    def test_sumsparse(self):

        indices = torch.tensor([[0, 1, 2, 0], [0, 1, 2, 1]], dtype=torch.long)
        vals = torch.tensor([1, 1, 1, 1], dtype=torch.float)

        print(vals / rgcn.sum_sparse(indices.t(), vals, (3, 3), row=False))





