import pdb
import sys
import torch
#import transformers
import skimage as ski

print("skimage", ski.__version__)

print("Python version: ", sys.version)
print("torch version: ", torch.__version__)
#print("transformers version: ", transformers.__version__)

t = torch.empty(3, 4, 5)
print("x,", t.size())
print(t.size())
print("t dim=-1:", t.size(dim=-1))