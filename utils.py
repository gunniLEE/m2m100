from distutils.log import INFO
import os
import time
import torch
import logging

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s,
                        datefmt='%m/%d/%Y %H:%M%S',
                        level=logging.INFO)

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]