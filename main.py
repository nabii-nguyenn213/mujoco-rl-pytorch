import os 
# NOTE : Comment these lines if run with MPI 
# os.environ.setdefault("OMP_NUM_THREADS", "8")
# os.environ.setdefault("MKL_NUM_THREADS", "8")
# os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
# os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import time 
import argparse
from utils.helper import loadConfig
from train.train_SAC import SAC
from utils.mpi_utils import get_rank, get_world_size

import torch
threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
torch.set_num_threads(threads)
torch.set_num_interop_threads(1)

def get_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    configDir = "configs/SAC.yaml" 
    config = loadConfig(configDir=configDir)
    if args.env is not None: 
        config["env"]["name"] = args.env
    start_time = time.perf_counter()
    if get_world_size() == 1: 
        exp = SAC(config)
    else: 
        exp = SAC(config, rank=get_rank())
    exp.run()
    end_time = time.perf_counter() - start_time
    if get_world_size() == 1:
        print(f"Total runtime : {end_time}s")
    else: 
        print(f"Total runtime {get_rank()} : {end_time}s")
