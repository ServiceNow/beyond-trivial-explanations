import numpy as np
from haven import haven_utils as hu
from src.datasets.biased_celeba import BiasedCelebA
from copy import deepcopy

ngpu = 1

EXP_GROUPS = {
    # 1. TRAIN a TCVAE on CELEBA
    "tcvae":  {"wrapper": "tcvae",
               # Hardware
               "ngpu": ngpu,
               "amp": 1,

               # Optimization
               "batch_size": 64 * ngpu,
               "target_loss": "val_loss",
               "lr": [0.0004],
               "max_epoch": 400,

               # Model
               "model": "biggan",
               "backbone": "resnet",
               "channels_width": 4,
               "z_dim": 128,
               "mlp_width": 4,
               "mlp_depth": 2,

               # TCVAE
               "beta": [0.001], # the idea is to be able to interpolate while getting good reconstructions
               "tc_weight": [1], # we keep the total_correlation penalty high to encourage disentanglement
               "vgg_weight": [1],
               "beta_annealing": [True],
               "dp_prob": 0.3,

               # Data
               "height": 128,
               "width": 128,
               "crop": 150,

               "dataset": "biased_celeba",
               "transforms": [["flip"]],
               "labels_path":   "celeba_meta_explanations/list_eval_unbiased.csv",
               "dataset_train": "celeba",
               "dataset_val":   "celeba",
               "dataset_test":  "celeba",
               },

    # 1.1. TRAIN a VAE on CELEBA (Optional for xGEM)
    "vae":  {"wrapper": "vae",
             # hardware
             "ngpu": ngpu,
             "amp": 1,

             # optimization
             "batch_size": 64 * ngpu,
             "target_loss": "val_loss",
             "lr": 0.0004,
             "max_epoch": 400,

             # model
             "model": "biggan",
             "backbone": "resnet",
             "channels_width": 4,
             "z_dim": 128,
             "mlp_width": 4,
             "mlp_depth": 2,

             # beta-vae
             "beta": 0.001,
             "vgg_loss": 0,  # set to 1 for xGEM+
             "beta_annealing": [True],
             "dp_prob": 0.3,

             # data
             "height": 128,
             "width": 128,
             "crop": 150,

             "dataset": "biased_celeba",
             "transforms": [["flip"]],
             "labels_path":   "celeba_meta_explanations/list_eval_unbiased.csv",
             "dataset_train": "celeba",
             "dataset_val":   "celeba",
             "dataset_test":  "celeba",

             },

    # 2.0 TRAIN THE ORACLE
    "cls_oracle":  {"wrapper": "classifier",
                    # Hardware
                    "ngpu": ngpu,
                    "amp": 0,

                    # Optimization
                    "batch_size": 32 * ngpu,
                    "target_loss": "val_loss",
                    "max_epoch": 5,
                    "optimizer": "sgd",

                    # Oracle
                    "lr": 0.0001,
                    "finetune_lr": [0],

                    # Model
                    "model": "vgg_face",
                    "n_classes": 40,

                    # Data
                    "height": 128,
                    "width": 128,
                    "crop": 150,

                    "labels_path": "celeba_meta_explanations/list_eval_unbiased.csv",
                    "dataset": "biased_celeba",
                    "transforms": ["flip"],
                    "dataset_train": "celeba",
                    "dataset_val":   "celeba",
                    "dataset_test":  "celeba",

                    },
    # 2.1 TRAIN THE CLASSIFIER TO BE FOOLED
    "unbiased_classifier":  {"wrapper": "classifier",
                             # Hardware
                             "ngpu": ngpu,
                             "amp": 0,

                             # Optimization
                             "batch_size": 32 * ngpu,
                             "target_loss": "val_loss",
                             "lr": [0.0001],
                             "finetune_lr": [1],
                             "max_epoch": [5],
                             "optimizer": "sgd",

                             # Model
                             "model": "densenet",
                             "n_classes": [40],
                             "dp_prob": 0.3,

                             # Data
                             "height": 128,
                             "width": 128,
                             "crop": 150,

                             "labels_path": "celeba_meta_explanations/list_eval_unbiased.csv",
                             "dataset": "biased_celeba",
                             "transforms": ["flip"],
                             "dataset_train": "celeba",
                             "dataset_val":   "celeba",
                             "dataset_test":  "celeba",

                             },

    # 3. Gender bias, to compare with Progressive Exaggeration
    "gender_biased_classifier":  {"wrapper": "classifier",
                                  # Hardware
                                  "ngpu": ngpu,
                                  "amp": 0,

                                  # Optimization
                                  "batch_size": 32*ngpu,
                                  "target_loss": "val_loss",
                                  "lr": [0.0001],
                                  "finetune_lr": 1,
                                  "early_stopping": 10,
                                  "max_epoch": 10,
                                  "optimizer": "sgd",

                                  # Model
                                  "model": "densenet",
                                  "backbone": "resnet",
                                  "n_classes": [1],
                                  "dp_prob": 0.3,

                                  # Data
                                  "height": 128,
                                  "width": 128,
                                  "crop": 150,

                                  "dataset": "biased_celeba",
                                  "transforms": ["flip"],
                                  "dataset_train": "celeba",
                                  "dataset_val":   "celeba",
                                  "dataset_test":  "celeba",
                                  "labels_path": "celeba_meta_explanations/list_eval_smiling_biased_gender.csv"

                                  },

    # 4. Gradient Attacks
    "gradient_attacks":  {"wrapper": "gradient_attacks",
                          # Hardware
                          "ngpu": ngpu,
                          "amp": 1,

                          # Optimization
                          "lr": 0.01,
                          "max_iters": 20,
                          "cache_batch_size": 64 * ngpu,
                          "force_cache": False,
                          "batch_size": 100 * ngpu,
                          # stop optimizing if x% of the counterfactuals are successful
                          "stop_batch_threshold": 0.9,
                          "seed": 42,

                          # Explanations config
                          "attribute": "Smiling",
                          "num_explanations": 8,
                          "method": "fisher spectral inv",
                          "reconstruction_weight": 10.,
                          "lasso_weight": 1.,
                          "diversity_weight": 1,
                          "n_samples": 10 * ngpu,
                          "fisher_samples": 0,

                          # Pretrained models and paths
                          "dataset_val": "celeba",
                          "labels_path": "celeba_meta_explanations/list_eval_unbiased.csv",
                          "generator_path": 'pretrained_models/dive',
                          "classifier_path": 'pretrained_models/unbiased_classifier',
                          "oracle_path": 'pretrained_models/oracle',
                          },
}


EXP_GROUPS = {k: hu.cartesian_exp_group(v) for k, v in EXP_GROUPS.items()}

# BEYOND TRIVIAL EXPLANATIONS
# First we need to compute the fisher information matrices
random_search = EXP_GROUPS['gradient_attacks'][0] # default hparams
methods = ['fisher_spectral', 'xgem+']
exps = []
n = 0
base_exp = deepcopy(random_search)
for method in methods:
    method_exp = deepcopy(base_exp)
    method_exp['cache_only'] = True
    method_exp['method'] = method
    if method_exp['method'] in 'xgem+':
        method_exp['generator_path'] = f'pretrained_models/{method}'
        if method_exp in exps:
            continue
    exps.append(method_exp)
EXP_GROUPS['cache_fim'] = exps

# For different methods (fisher spectral, random, dive, xgem)
# Explore random hyperparameter combinations. 
# Launches a lot of jobs but they run fast on single gpu
np.random.seed(42)
random_search = EXP_GROUPS['gradient_attacks'][0] # default hparams
n_trials = 4
lasso_space = [0] + list([0.0001, 0.001, 0.01, 0.1, 1, 10])
lr_space = [0.005, 0.01, 0.05, 0.1]
diversity_space = [0] + list([0.001, 0.01, 0.1, 1, 10])
reconstruction_space = [0] + list([0.0001, 0.001, 0.01, 0.1, 1, 10])
num_trials = list(range(2, 16))
fisher_samples = list([10])
methods = ['fisher_spectral', 'random', 'dive', 'xgem+']
exps = []
n = 0
all_attributes = [BiasedCelebA.all_attributes[i]
                  for i in np.random.permutation(len(BiasedCelebA.all_attributes))]
for run in range(n_trials):
    for attribute in all_attributes:
        base_exp = deepcopy(random_search)
        base_exp['lr'] = float(np.random.choice(lr_space))
        base_exp['lasso_weight'] = float(np.random.choice(lasso_space))
        base_exp['diversity_weight'] = float(np.random.choice(diversity_space))
        base_exp['reconstruction_weight'] = float(
            np.random.choice(reconstruction_space))
        base_exp['num_explanations'] = int(np.random.choice(num_trials))
        base_exp['attribute'] = attribute
        for method in methods:
            method_exp = deepcopy(base_exp)
            method_exp['method'] = method
            if method_exp['method'] in 'xgem+':
                method_exp['lasso_weight'] = 0
                method_exp['diversity_weight'] = 0
                method_exp['generator_path'] = f'pretrained_models/{method}'
                if method_exp in exps:
                    continue
            exps.append(method_exp)
EXP_GROUPS['random_search'] = exps
