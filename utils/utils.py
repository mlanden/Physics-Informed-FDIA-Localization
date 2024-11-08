import json
import torch
import torch.multiprocessing as mp
from typing import List
import queue
import traceback
from sklearn.metrics import roc_curve
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt



def to_complex(s: str):
    if len(s) == 0:
        return complex(0)
    else:
        s = s.replace(" ", "")
        s = s.replace("(", "")
        s = s.replace(")", "")
        if "j" in s:
            s = s.replace("j", "") + "j"
        s = s.replace("i", "j")
        return complex(s)
        