import resource
import torch
import time

def clean_cuda(*wargs, **kwargs):
    """all variable neet to del"""
    for x in wargs: del x
    for k in kwargs: del kwargs[k]

    with torch.no_grad():
        torch.cuda.empty_cache()
 

def mem_usage():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return f'mem usage= --->{usage[2]/1024.0} mb'