import base64
import hashlib
import os
import sys
import time
from copy import deepcopy
from json import decoder
from os.path import join

import cv2
import h5py
import numpy as np
import pylab
import pylab as pl
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from haven import haven_utils as hu
from numpy.lib.function_base import vectorize
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from src import datasets, models, wrappers
from src.models.biggan import Decoder
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


class DatasetWrapper(torch.utils.data.Dataset):
    """Helper class to provide image id"""

    def __init__(self, dataset, indices=None):
        """Constructor

        Args:
            dataset (torch.utils.data.Dataset): Dataset object
        """
        self.dataset = dataset
        self.indices = indices
        if self.indices is None:
            self.indices = list(range(len(dataset)))

    def __getitem__(self, item):
        return (self.indices[item], *self.dataset[self.indices[item]])

    def __len__(self):
        return len(self.indices)


class GradientAttack(torch.nn.Module):
    """Main class to generate counterfactuals"""

    def __init__(self, exp_dict, savedir, data_path):
        """Constructor

        Args:
            exp_dict (dict): hyperparameter dictionary
            savedir (str): root path to experiment directory
            data_path (str): root path to datasets and pretrained models
        """
        super().__init__()
        self.exp_dict = exp_dict
        self.savedir = savedir
        self.generator_path = os.path.join(
            data_path, self.exp_dict["generator_path"])
        self.classifier_path = os.path.join(
            data_path, self.exp_dict["classifier_path"])
        self.oracle_path = os.path.join(data_path, self.exp_dict["oracle_path"])
        self.generator_dict = hu.load_json(
            os.path.join(self.generator_path, 'exp_dict.json'))
        self.classifier_dict = hu.load_json(
            os.path.join(self.classifier_path, 'exp_dict.json'))
        self.oracle_dict = hu.load_json(
            os.path.join(self.oracle_path, 'exp_dict.json'))
        self.dataset = DatasetWrapper(datasets.get_dataset(
            data_path, 'val', self.classifier_dict))
        self.current_attribute = self.dataset.dataset.all_attributes.index(
            self.exp_dict['attribute'])
        self.data_path = data_path
        self.ngpu = exp_dict['ngpu']
        self.exp_dict = exp_dict
        self.load_oracle()
        self.load_models()

        if self.exp_dict.get("cache_only", False):
            self.read_or_write_cache()
        else:
            self.read_or_write_cache()
            self.select_data_subset()
            self.attack_dataset()

    def get_loader(self, batch_size):
        """Helper function to create a dataloader

        Args:
            batch_size (int): the batch_size

        Returns:
            torch.utils.data.DataLoader: dataloader without shuffling and chosen bs
        """
        return torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                           num_workers=4, drop_last=False, shuffle=False)

    def read_or_write_cache(self):
        """Compute and store the Fisher Information Matrix 
            for future use as well as other auxiliary data

        Raises:
            TimeoutError: if two processes collide, 
                one will wait for the other to finish 
                to avoid repeating operations. 
                If the wait is too long, it raises an error.
        """
        self.digest = f"cache_{hu.hash_dict(self.classifier_dict)}_{hu.hash_dict(self.generator_dict)}"
        self.digest = os.path.join(self.generator_path, self.digest)

        if self.exp_dict["force_cache"] == True:
            os.remove(self.digest)

        if not os.path.isfile(self.digest):
            try:
                os.makedirs(f"{self.digest}.lock")
                lock = False
            except FileExistsError:
                lock = True
            if lock:
                print(
                    f"Waiting for another process to finish computing cache or delete {self.digest}.lock to continue")
                t = time.time()
                while os.path.isdir(f"{self.digest}.lock"):
                    if (time.time() - t) / 3600 > 2:
                        raise TimeoutError(f"Timout while waiting for another process to \
                                            finish computing cache on {self.digest}. Delete \
                                            if that is not the case")
                    time.sleep(1)
                lock = False
            self.write_cache()

        self.read_cache()

    def write_cache(self):
        """Loops through the data and stores latents and FIM"""

        print("Caching FIM")
        self.generative_model.exp_dict['amp'] = 0
        loader = self.get_loader(batch_size=self.exp_dict['cache_batch_size'])
        mus = []
        logvars = []
        reconstructions = []
        preds = []
        fishers = []
        fishers = 0
        for idx, x, y in tqdm(loader):
            with torch.no_grad():
                x = x.cuda()
                labels = y.cuda()
                mu, logvar = torch.nn.DataParallel(
                    self.generative_model.encoder, list(range(self.ngpu)))(x).chunk(2, 1)
                mus.append(mu.cpu())
                logvars.append(logvar.cpu())
                b, c = mu.size()
                z = mu.data.clone()
                # z.requires_grad = False
                self.first = True

            # self.generative_model.zero_grad()
            # self.classifier.zero_grad()
            # z.grad = None
            def jacobian_forward(z):
                reconstruction = torch.nn.DataParallel(
                    self.generative_model.decoder, list(range(self.ngpu)))(z)
                logits = torch.nn.DataParallel(self.classifier, list(range(self.ngpu)))(
                    reconstruction)  # [:, self.current_attribute, None]
                if self.first:
                    preds.append(logits.data.cpu().numpy())
                    self.first = False
                y = torch.distributions.Bernoulli(
                    logits=logits).sample().detach()
                logits = logits * y + (1 - logits) * (1 - y)
                loss = logits.sum(0)
                return loss
            grads = torch.autograd.functional.jacobian(jacobian_forward, z)
            with torch.no_grad():
                fisher = torch.matmul(grads[:, :, :, None], grads[:, :, None, :]).view(
                    40, b, c, c).sum(1).cpu()
            fishers += fisher.numpy()
            del(fisher)
            del(z)

        with h5py.File(self.digest, 'w') as outfile:
            to_save = dict(
                fisher=fishers,
                mus=np.concatenate(mus, 0),
                logvars=np.concatenate(logvars, 0),
                logits=np.concatenate(preds, 0),
            )
            for k, v in to_save.items():
                outfile[k] = v
        os.removedirs(f"{self.digest}.lock")
        self.generative_model.exp_dict['amp'] = self.exp_dict['amp']
        print("Done.")

    def read_cache(self):
        """Reads cached data from disk"""
        print("Loading from %s" % self.digest)
        self.loaded_data = h5py.File(self.digest, 'r')
        try:
            self.fisher = torch.from_numpy(self.loaded_data['fisher'][...])
            self.logits = torch.from_numpy(self.loaded_data['logits'][...])
            self.mus = torch.from_numpy(self.loaded_data['mus'][...])
        except:
            print("FIM not found in the hdf5 cache")
        print("Done.")

    def load_generator(self):
        """Helper function to load the generator model"""

        print("Loading generator...")
        self.generator_dict['amp'] = self.exp_dict['amp']
        self.generative_model = models.get_model(
            self.generator_dict["model"], self.generator_dict)
        try:
            self.generative_model.load_state_dict(torch.load(
                os.path.join(self.generator_path, 'checkpoint.pth'))['model'])
        except:
            from collections import OrderedDict
            data = torch.load(os.path.join(
                self.generator_path, 'checkpoint.pth'))['model']
            data = OrderedDict([(k.replace("model.", ""), v)
                               for k, v in data.items() if k.split(".")[0] == 'model'])
            self.generative_model.load_state_dict(data)
        self.generative_model.eval().cuda()

    def load_classifier(self):
        """Helper function to load the classifier model"""

        print("Loading classifier...")
        wrapper = wrappers.get_wrapper(
            self.classifier_dict['wrapper'], self.classifier_dict, '', '')
        classifier = wrapper.classifier
        classifier.load_state_dict(torch.load(os.path.join(
            self.classifier_path, 'checkpoint.pth'))['classifier'])
        feat_extract = wrapper.feat_extract
        feat_extract.load_state_dict(torch.load(os.path.join(
            self.classifier_path, 'checkpoint.pth'))['feat_extract'])
        self.classifier = torch.nn.Sequential(*[feat_extract, classifier])
        self.classifier.eval().cuda()
        self.n_classes = self.classifier_dict["n_classes"]
        print("Done")

    def load_oracle(self):
        """Helper function to load the oracle model"""

        print("Loading oracle...")
        self.oracle_dict['wrapper'] = 'classifier'
        self.oracle_dict['ngpu'] = 1
        wrapper = wrappers.get_wrapper(self.oracle_dict['wrapper'], self.oracle_dict,
                                       '', self.data_path)
        oracle_classifier = wrapper.classifier
        oracle_classifier.load_state_dict(torch.load(
            os.path.join(self.oracle_path, 'checkpoint.pth'))['classifier'])
        oracle_extractor = wrapper.feat_extract
        oracle_extractor.load_state_dict(torch.load(os.path.join(
            self.oracle_path, 'checkpoint.pth'))['feat_extract'])

        class OracleClassifier(torch.nn.Module):
            def __init__(self, feat_extract, classifier):
                super().__init__()
                self.feat_extract = feat_extract
                self.classifier = classifier

            def forward(self, x):
                f = self.feat_extract(x)
                return f, self.classifier(f)
        self.oracle = OracleClassifier(oracle_extractor, oracle_classifier)
        self.oracle.eval().cuda()
        self.oracle = torch.nn.DataParallel(
            self.oracle, list(range(self.exp_dict['ngpu'])))
        print("Done")

    def load_models(self):
        """Helper function to load everything"""
        self.load_generator()
        self.load_classifier()

    def load_dataset(self, split="train"):
        """Helper function to load a dataset used in a previous experiment

        Args:
            split (str, optional): Defaults to "train".
        """
        print("Loading dataset..")
        self.dataset = DatasetWrapper(datasets.get_dataset(
            self.data_path, split, self.classifier_dict))
        setattr(self, "dataset_%s" % split, self.dataset)
        print("Done")

    def select_data_subset(self):
        """Instead of using the whole dataset, we use a balanced set of correctly and incorrectly classified samples"""
        if self.exp_dict["n_samples"] > 0:
            preds = torch.sigmoid(self.logits[..., self.current_attribute]).numpy()
            labels = self.dataset.dataset.y[..., self.current_attribute].astype(float)
            indices = []
            for confidence in [-0.9, -0.6, -0.4, -0.1, 0.1, 0.4, 0.6, 0.9]:
                # obtain samples that are closest to the required level of confidence
                indices.append(np.abs(labels - preds - confidence).argsort()
                               [:self.exp_dict["n_samples"]])
            indices = np.concatenate(indices, 0)
            self.dataset.indices = indices

    def get_mask(self, batch, latents):
        """Helper function that outputs a binary mask for the latent 
            space during the counterfactual explanation 

        Args:
            latents (torch.Tensor): dataset latents (precomputed)

        Returns:
            torch.Tensor: latents mask
        """
        method = self.exp_dict["method"]
        num_explanations = self.exp_dict['num_explanations']

        if 'fisher' in method:
            if self.exp_dict['fisher_samples'] <= 0:
                fishers = [self.fisher[self.current_attribute]]
            else:
                fishers = self.get_pointwise_fisher(batch, self.exp_dict['fisher_samples'])

        if method in ["fisher_chunk"]:
            masks = []
            for fisher in fishers:
                indices = torch.diagonal(fisher).argsort(descending=True, dim=-1)
                mask = torch.ones(
                    num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
                chunk_size = latents.shape[1] // num_explanations
                for i in range(num_explanations):
                    mask.data[i, indices[(i * chunk_size)
                                        :((i + 1) * chunk_size)]] = 0
                masks.append(mask)
            mask = torch.stack(masks, 0)
        elif method in ["fisher_range"]:
            masks = []
            for fisher in fishers:
                indices = torch.diagonal(fisher).argsort(descending=True, dim=-1)
                mask = torch.ones(
                    num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
                for i in range(num_explanations):
                    mask.data[i, indices[0:i]] = 0
                masks.append(mask) 
            mask = torch.stack(masks, 0)
        elif method in ["fisher_spectral", "fisher_spectral_inv"]:
            masks = []
            for fisher in fishers:
                scluster = SpectralClustering(n_clusters=num_explanations, affinity='precomputed',
                                            assign_labels='discretize', random_state=0, eigen_solver='arpack', eigen_tol=1e-6)
                affinity = fisher.numpy()
                affinity = affinity - affinity.min()
                affinity /= affinity.max()
                scluster.fit(affinity)
                mask = torch.zeros(
                    num_explanations, latents.shape[1], device=latents.device)
                for i in range(num_explanations):
                    mask[i, torch.from_numpy(scluster.labels_).to(
                        latents.device) == i] = 1
                masks.append(mask)
            mask = torch.stack(masks, 0)
            if 'inv' in method:
                mask = 1 - mask
        elif method in ["random"]:
            indices = torch.randperm(latents.shape[1], device=latents.device)
            mask = torch.ones(
                num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
            chunk_size = latents.shape[1] // num_explanations
            for i in range(num_explanations):
                mask.data[i, indices[(i * chunk_size):((i + 1) * chunk_size)]] = 0
            mask = mask[None, ...]
        else:
            mask = torch.ones(1,
                num_explanations, latents.shape[1], device=latents.device, dtype=latents.dtype)
        return mask

    def attack_batch(self, batch):
        """Uses gradient descent to compute counterfactual explanations

        Args:
            batch (tuple): a batch of image ids, images, and labels

        Returns:
            dict: a dictionary containing the whole attack history
        """
        idx, images, labels = batch
        idx = idx.cuda(non_blocking=True)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        predicted_labels = torch.sigmoid(self.logits[idx])
        predicted_labels = predicted_labels[:,
                                            self.current_attribute, None] > 0.5
        predicted_labels = predicted_labels.float().cuda()

        diversity_weight = self.exp_dict['diversity_weight']
        lasso_weight = self.exp_dict['lasso_weight']
        reconstruction_weight = self.exp_dict['reconstruction_weight']

        latents = self.mus[idx].cuda()
        b, c = latents.size()

        num_explanations = self.exp_dict['num_explanations']
        epsilon = torch.randn(b, num_explanations, c,
                              requires_grad=True,
                              device=latents.device)
        epsilon.data *= 0.01

        mask = self.get_mask(batch, latents)

        optimizer = torch.optim.Adam([epsilon],
                                     lr=self.exp_dict['lr'],
                                     weight_decay=0)
        if self.exp_dict["amp"] > 0:
            scaler = amp.GradScaler()
        attack_history = []

        class DecoderClassifier(torch.nn.Module):
            def __init__(self, g, c):
                super().__init__()
                self.g = g
                self.c = c

            def forward(self, x):
                decoded = self.g(x)
                return decoded, self.c(decoded)

        decoder_classifier = DecoderClassifier(
            self.generative_model.decoder, self.classifier)
        decoder_classifier = torch.nn.DataParallel(
            decoder_classifier, list(range(self.ngpu)))

        opt_mask = torch.ones(epsilon.shape[0], epsilon.shape[1], device=epsilon.device, dtype=torch.float)

        for it in range(self.exp_dict['max_iters']):
            optimizer.zero_grad()
            div_regularizer = 0
            lasso_regularizer = 0
            reconstruction_regularizer = 0

            epsilon.data = epsilon.data * mask
            z_perturbed = latents[:, None, :].detach() + epsilon
            with amp.autocast(enabled=self.exp_dict['amp'] > 0):
                if diversity_weight > 0:
                    epsilon_normed = epsilon * opt_mask[:, :, None]
                    epsilon_normed = F.normalize(epsilon_normed, 2, -1)
                    div_regularizer = torch.matmul(
                        epsilon_normed, epsilon_normed.permute(0, 2, 1))
                    div_regularizer = div_regularizer * (1 - torch.eye(div_regularizer.shape[-1],
                                                                       dtype=div_regularizer.dtype,
                                                                       device=div_regularizer.device))[None, ...]
                    div_regularizer = (div_regularizer ** 2).sum()

                decoded, logits = decoder_classifier(
                    z_perturbed.view(b * num_explanations, c))
                bn, ch, h, w = decoded.size()
                if reconstruction_weight > 0:
                    reconstruction_regularizer = (torch.abs(
                        images[:, None, ...] - decoded.view(b, num_explanations, ch, h, w)) * \
                            opt_mask.view(b, num_explanations, 1, 1, 1)).mean((3,4)).sum()

                lasso_regularizer = (torch.abs(
                    z_perturbed - latents[:, None, :]) * opt_mask[:, :, None]).sum()

                regularizer = lasso_regularizer * lasso_weight + \
                    div_regularizer * diversity_weight + \
                    reconstruction_regularizer * reconstruction_weight
                regularizer = regularizer / mask.expand_as(z_perturbed).sum()

                loss_attack = F.binary_cross_entropy_with_logits(logits[:, self.current_attribute, None],
                                                                 1 - predicted_labels[:, None, :].repeat(1, num_explanations, 1).view(b * num_explanations, 1),
                                                                 reduction='none')
                loss_attack = (loss_attack.view(b, num_explanations) * opt_mask).mean()
                loss = loss_attack + regularizer
            if self.exp_dict["amp"] > 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            attack_history = dict(
                iter=np.array([it]),
                idx=idx.data.cpu().numpy(),
                logits=logits.data.cpu().view(b, num_explanations, -1).numpy(),
                labels=labels.data.cpu().numpy().argmax(-1),
                loss=np.array([float(loss)]),
                loss_attack=np.array([float(loss_attack)]),
                reconstruction_regularizer=np.array(
                    [float(reconstruction_regularizer)]),
                div_regularizer=np.array([float(div_regularizer)]),
                lasso_regularizer=np.array([float(lasso_regularizer)]),
            )  # instead of append, directly set dict due to memory constraints
            success_rate = float(((torch.sigmoid(logits[:, self.current_attribute].view(b, -1)) > 0.5) !=
                                  (predicted_labels.view(b, 1) == 1)).float().mean())
            attack_history['success_rate'] = np.array([success_rate])
            # opt_mask = (logits[:, self.current_attribute].view(b, num_explanations) > 1) == \
            # opt_mask = opt
            if success_rate >= self.exp_dict["stop_batch_threshold"]:
                break

        with torch.no_grad():
            with amp.autocast(enabled=self.exp_dict['amp'] > 0):
                torch.cuda.empty_cache()
                f_attack, oracle_preds_attack = self.oracle(decoded)
                f, oracle_preds_reconstruction = self.oracle(images)
                attack_history['oracle_preds_attack'] = oracle_preds_attack.data.cpu().view(
                    b, num_explanations, -1).numpy()
                attack_history['oracle_preds_reconstruction'] = oracle_preds_reconstruction.data.cpu(
                ).view(b, 1, -1).numpy()
                attack_history['latent_similarity'] = (F.normalize(f, 2, -1)[:, None, :] *
                                                       F.normalize(f_attack, 2, -1).view(b, self.exp_dict['num_explanations'], -1)).sum(-1).view(b, -1).data.cpu().numpy()

        return attack_history

    def attack_dataset(self):
        """Loops over the dataset and generates counterfactuals for all the samples"""

        loader = self.get_loader(
            self.exp_dict['batch_size'] // self.exp_dict['num_explanations'])
        attack_histories = None
        for batch in tqdm(loader):
            history = self.attack_batch(batch)
            if attack_histories is None:
                attack_histories = {k: [v] for k, v in history.items()}
            else:
                for k, v in history.items():
                    attack_histories[k].append(v)
            print(f"total_loss: {history['loss']},",
                  f"loss_attack: {history['loss_attack']},",
                  f"reconstruction: {history['reconstruction_regularizer']},",
                  f"lasso: {history['lasso_regularizer']},",
                  f"diversity: {history['div_regularizer']},",
                  f"success_rate: {history['success_rate']}")
        with h5py.File(os.path.join(self.savedir, 'results.h5'), 'w') as outfile:
            for k, v in attack_histories.items():
                print(f'saving {k}')
                outfile[k] = np.concatenate(v, 0)

    def get_pointwise_fisher(self, batch, num_samples):
        idx, x, y = batch
        with torch.no_grad():
            x = x.cuda()
            labels = y.cuda()
            mu, logvar = torch.nn.DataParallel(
                self.generative_model.encoder, list(range(self.ngpu)))(x).chunk(2, 1)
            b, c = mu.size()
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(num_samples, mu.shape[0], mu.shape[1], device=mu.device)
            z = eps * std[None, ...] + mu[None, ...]
            z = z.view(num_samples, b, c)
            # z.requires_grad = False

        # self.generative_model.zero_grad()
        # self.classifier.zero_grad()
        # z.grad = None
        def jacobian_forward(z):
            reconstruction = torch.nn.DataParallel(
                self.generative_model.decoder, list(range(self.ngpu)))(z)
            logits = torch.nn.DataParallel(self.classifier, list(range(self.ngpu)))(
                reconstruction)[:, self.current_attribute, None]
            y = torch.distributions.Bernoulli(
                logits=logits).sample().detach()
            logits = logits * y + (1 - logits) * (1 - y)
            return logits.sum()
        fisher = 0
        for i in range(num_samples):
            grads = torch.autograd.functional.jacobian(jacobian_forward, z[i])
            with torch.no_grad():
                fisher += torch.matmul(grads[:, :, None], grads[:, None, :]).cpu()
            return fisher