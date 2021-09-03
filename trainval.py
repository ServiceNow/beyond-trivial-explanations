import argparse
import os
import shutil as sh

import numpy as np
import pandas as pd
import torch
from haven import haven_utils as hu
from haven import haven_wizard as hw
from PIL import Image

import exp_configs
from src import datasets, wrappers

torch.backends.cudnn.benchmark = True


def trainval(exp_dict, savedir, args):
    # ==========================
    # load datasets
    # ==========================
    train_set = datasets.get_dataset(args.dataroot, 'train', exp_dict)
    val_set = datasets.get_dataset(args.dataroot, 'val', exp_dict)

    # ==========================
    # get dataloaders
    # ==========================
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=exp_dict["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True)

    # ==========================
    # create model and trainer
    # ==========================
    model = wrappers.get_wrapper(exp_dict["wrapper"],
                                 exp_dict=exp_dict,
                                 savedir=savedir,
                                 datadir=args.dataroot)

    model_path = os.path.join(savedir, "checkpoint.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        model.load_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Run training and validation
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        # Train
        train_dict = model.train_on_loader(epoch, train_loader)

        # Validate
        val_dict = model.val_on_loader(epoch, val_loader, vis_flag=True)

        # Visualize Results
        if 'val_images' in val_dict:
            path = os.path.join(savedir, 'images', "reconstruction.png")
            if os.path.isfile(path):
                path_old = os.path.join(
                    savedir, 'images', "previous_reconstruction.png")
                sh.move(path, path_old)
            Image.fromarray(val_dict['val_images']).save(path, "PNG")
            del val_dict['val_images']

        # Create score_dict
        score_dict = {"epoch": epoch}
        score_dict.update(model.get_lr())
        score_dict.update(train_dict)
        score_dict.update(val_dict)

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())

        # Save checkpoint
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(model_path, model.get_state_dict())
        print("Saved: %s" % savedir)

        # Save best checkpoint
        if exp_dict.get('early_stopping', False):
            if score_dict['val_loss'] <= score_df['val_loss'][:-1].min():
                hu.save_pkl(os.path.join(
                    savedir, "score_list_best.pkl"), score_list)
                hu.torch_save(os.path.join(
                    savedir, "checkpoint_best.pth"), model.get_state_dict())
                print("Saved Best: %s" % savedir)
            # Check for end of training conditions
            elif exp_dict.get('early_stopping') < (epoch - score_df['val_loss'].argmin()):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+",
                        help="Name of the experiments to run")
    parser.add_argument('-sb', '--savedir_base', required=True,
                        help="The experiment will be backup and run in this folder.")
    parser.add_argument('-nw', '--num_workers', type=int, default=0,
                        help="DataLoader number of workers")
    parser.add_argument('-dr', '--dataroot', type=str, default="./data",
                        help="Path to datasets and pretrained models")
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--run_jobs", type=str, default="0", 
                        help="Whether to run in cluster mode")
    parser.add_argument("-p", "--python_binary", type=str, required=True,
                        help="path to python (e.g. /usr/bin/python)")

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
