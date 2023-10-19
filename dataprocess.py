
import numpy as np
import torch
from pkg_resources import packaging

print("Torch version:", torch.__version__)

import clip

clip.available_models()


AAA = clip.tokenize("The drug response value between CC1(CC2=C(C(=O)C1)C(=NN2C3=CC(=C(C=C3)C(=O)N)NC4CCC(CC4)O)C(F)(F)F)C and CAL-29 is 0.59")
print(AAA)
BBB = clip.tokenize("The drug response value between CC1(CC2=C(C(=O)C1)C(=NN2C3=CC(=C(C=C3)C(=O)N)NC4CCC(CC4)O)C(F)(F)F)C and CAL-21 is 0.31")
print(BBB)

print(AAA-BBB)
