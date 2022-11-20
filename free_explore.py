# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:09:18 2022

@author: 51027
"""

import galai as gal
import torch

torch.cuda.is_available()
torch.cuda.current_device()
torch.cuda.device_count()
torch.cuda.get_device_name(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


model = gal.load_model("mini", num_gpus=1)
model.generate("Scaled dot product attention:\n\n\\[")
# Scaled dot product attention:\n\n\\[ \\displaystyle\\text{Attention}(Q,K,V)=\\text{softmax}(\\frac{QK^{T}}{\\sqrt{d_{k}}}%\n)V \\]


torch.__version__ # Get PyTorch and CUDA version
torch.cuda.is_available() # Check that CUDA works
torch.cuda.device_count() # Check how many CUDA capable devices you have

