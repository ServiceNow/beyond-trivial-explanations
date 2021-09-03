import torch
import argparse
import pandas as pd
import os
import torch
import exp_configs
import shutil as sh
import numpy as np

from haven import haven_utils as hu
from haven import haven_wizard as hw
from src import wrappers
from src import datasets
from PIL import Image

torch.backends.cudnn.benchmark = True

def trainval(exp_dict, savedir, args):
    model = wrappers.get_wrapper(exp_dict["wrapper"],  
                                 exp_dict=exp_dict, 
                                 savedir=savedir, 
                                 datadir=args.dataroot)

    pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+", 
                            help="Name of the experiments to run")
    parser.add_argument('-sb', '--savedir_base', required=True, 
                            help="The experiment will be backup and run in this folder.")
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-dr', '--dataroot', type=str, default="./data")
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs")
    parser.add_argument("-p", "--python_binary")

    args, unknown = parser.parse_known_args()

    # Get job config to run things on cluster
    job_config = None
    if os.path.exists('job_configs.py'):
        import job_configs
        job_config = job_configs.JOB_CONFIG
    
    # Run trinval function for each experiment
    hw.run_wizard(func=trainval, 
                exp_groups=exp_configs.EXP_GROUPS, 
                job_config=job_config,
                python_binary_path=args.python_binary,
                use_threads=True,
                args=args, 
                results_fname='results.ipynb')