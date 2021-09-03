import os

import numpy as np
import pylab
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
from src.models import get_model
from src.wrappers.base_wrapper import BaseWrapper
from tqdm import tqdm

from .modules.tcvae import btc_vae_loss
from .modules.vae import get_kl_loss
from .modules.vgg_loss import VGGPerceptualLoss


class TCVAE(BaseWrapper):
    """Trains a model on multiple GPUs"""

    def __init__(self, exp_dict, savedir, datadir):
        """ Constructor
        Args:
            model: architecture to train
            exp_dict: reference to dictionary with the global state of the application
            savedir: where to save images for debugging
            writer: tensorboard 
        """
        super().__init__()
        # Create model, opt, wrapper
        model = get_model(exp_dict["model"], exp_dict=exp_dict)
        self.model = model.cuda()
        self.exp_dict = exp_dict
        self.ngpu = self.exp_dict["ngpu"]
        self.devices = list(range(self.ngpu))
        self.savedir = savedir
        self.beta = self.exp_dict["beta"]
        if self.exp_dict["vgg_weight"] > 0:
            self.perceptual_loss = VGGPerceptualLoss(resize=False).cuda()
        self.model_parallel = torch.nn.DataParallel(
            self.model, list(range(self.ngpu)))
        # self.discriminator = Discriminator(ratio=self.model.ratio,
        #                                    width=self.exp_dict["channels_width"],
        #                                    dp_prob=exp_dict["dp_prob"]).cuda()
        # self.discriminator_loss = DiscriminatorLoss(self.discriminator)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          betas=(0.9, 0.999),
                                          lr=self.exp_dict["lr"])
        if self.exp_dict["amp"] > 0:
            self.scaler = amp.GradScaler()

    # def save_img(self, name, images, idx=0):
    #     """Helper function to save images

    #     Args:
    #         name (str): Image name
    #         images (list of np array): List of images to save concatenated horizontally
    #         idx (int, optional): iteration when the image was saved. Defaults to 0.
    #     """
    #     # im_orig = x_orig.permute(1,2,0).data.cpu().numpy()
    #     # im_reco = x_reco.permute(1,2,0).data.cpu().numpy()
    #     # im = np.concatenate([im_orig, im_reco], 1)
    #     # im = np.floor((im + 1) * 0.5 * 255)
    #     #pylab.imsave(os.path.join(self.savedir, 'tmp_%d.jpg' %idx), im.astype('uint8'))
    #     for im in images:
    #         im[...] = (im - im.min()) / (im.max() - im.min())
    #     im = torch.cat(images, 2)
    #     self.writer.add_image(name, im, idx)

    @torch.no_grad()
    def save_img(self, name, images, idx=0):
        """Helper function to save images

        Args:
        name (str): Image name
        images (list of np array): List of images to save concatenated horizontally
        idx (int, optional): iteration when the image was saved. Defaults to 0.
        """
        # im_orig = x_orig.permute(1,2,0).data.cpu().numpy()
        # im_reco = x_reco.permute(1,2,0).data.cpu().numpy()
        # im = np.concatenate([im_orig, im_reco], 1)
        # im = np.floor((im + 1) * 0.5 * 255)
        #pylab.imsave(os.path.join(self.savedir, 'tmp_%d.jpg' %idx), im.astype('uint8'))
        im = torch.cat(images, 2).data
        im = (im + 1) / 2
        im = im.permute(1, 2, 0)
        im = im * 255
        im = im.cpu().numpy().astype('uint8')
        return im

    def get_beta(self, epoch):
        """Helper function for beta annealing in beta-vae
        Fu, Hao, et al. "Cyclical annealing schedule: A simple approach to mitigating kl vanishing." arXiv preprint arXiv:1903.10145 (2019).

        Args:
            epoch (int): Current training epoch

        Returns:
            float: beta
        """
        if self.exp_dict["beta"] > 0:
            if self.exp_dict["beta_annealing"] == True:
                cycle_size = (self.exp_dict["max_epoch"] // 4)
                _epoch = epoch % cycle_size
                ratio = max(self.beta * 1e-3,
                            min(1, _epoch / (cycle_size * 0.5)))
                beta = self.beta * ratio
            else:
                beta = self.beta
        else:
            beta = 0
        return beta

    def train_vae_on_batch(self, epoch, batch_idx, batch):
        """Runs one batch iteration

        Args:
            epoch (int): Current epoch
            batch_idx (int): Current batch in epoch
            batch (tuple(Tensor)): Batch with images and labels
            vis_flag (bool): Whether to visualize images or not

        Returns:
            dict: Dictionary with the training metrics
        """
        x, y = batch
        b, c, h, w = x.size()
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        self.model.zero_grad()

        with amp.autocast(enabled=self.exp_dict["amp"] > 0):
            b = x.size(0)
            if self.ngpu > 1:
                mu, logvar, z, reconstruction = self.model_parallel(x)
            else:
                mu, logvar, z, reconstruction = self.model(x)

            kl_loss = 0
            kl_loss = get_kl_loss(mu, logvar)

            # Encoder
            b = x.size(0)
            if self.exp_dict["vgg_weight"] > 0:
                with amp.autocast(enabled=False):
                    vgg_mse = torch.nn.DataParallel(self.perceptual_loss, list(
                        range(self.ngpu)))(reconstruction, x).mean()
                pix_mse = 0
            else:
                pix_mse = l1_loss(x, reconstruction)
                vgg_mse = 0
            loss = vgg_mse + pix_mse
            beta = self.get_beta(epoch)
            if self.exp_dict["tc_weight"] > 0 and beta > 0:
                mi_loss, tc_loss, dw_kl_loss = btc_vae_loss(self.n_data, (mu, logvar),
                                                            latent_sample=z, is_mss=True)
                loss += mi_loss * self.beta + tc_loss * \
                    self.exp_dict["tc_weight"] * self.beta + dw_kl_loss * beta
            elif beta > 0:
                loss += kl_loss * beta
        if self.exp_dict["amp"] > 0:
            loss = self.scaler.scale(loss)
            loss.backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return dict(train_loss=float(kl_loss + pix_mse),
                    pix_mse=float(pix_mse),
                    vgg_mse=float(vgg_mse),
                    kl_loss=float(kl_loss),
                    tc_loss=float(tc_loss),
                    tc_weight=float(self.exp_dict["tc_weight"]),
                    running_beta=float(beta),
                    mean_mu=float(mu.mean()),
                    mean_logvar=float(logvar.mean()))

    def val_on_batch(self, epoch, batch_idx, batch, vis_flag):
        """Validate on one batch

        Args:
            epoch (int): current epoch
            batch_idx (int): current batch
            batch (tuple): images and labels
            vis_flag (bool): whether to visualize reconstructions

        Returns:
            dict: metrics
        """
        x, y = batch
        b = x.size(0)
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        mu, logvar, z, reconstruction = self.model_parallel(x)
        kl_loss = get_kl_loss(mu, logvar)
        pix_mse_loss = l1_loss(x, reconstruction)

        ret = dict(val_kl_loss=float(kl_loss),
                   val_pix_mse_loss=float(pix_mse_loss),
                   val_loss=float(pix_mse_loss + kl_loss),
                   mu=mu.data.cpu().numpy())

        if vis_flag and batch_idx == 0:
            with amp.autocast(enabled=self.exp_dict["amp"] > 0):
                z_2 = torch.randn_like(z)
                reconstruction_2 = self.model.decode(z_2)
            im = self.save_img("val_reconstruction", [
                x[0], reconstruction[0], reconstruction_2[0]], epoch)
            ret['val_images'] = im
        return ret

    def predict_on_batch(self, x):
        if self.ngpu > 1:
            return self.model_parallel(x.cuda())
        else:
            return self.model(x.cuda())

    def train_on_loader(self, epoch, data_loader):
        """Iterate over the training set

        Args:
            data_loader: iterable training data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.train()
        self.n_data = len(data_loader.dataset)
        ret = {}
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            res_dict = self.train_vae_on_batch(epoch, batch_idx, batch)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        torch.cuda.empty_cache()
        return {k: np.mean(v) for k, v in ret.items()}

    @torch.no_grad()
    def val_on_loader(self, epoch, data_loader, vis_flag=False):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        ret = {}
        labels = []
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            x, y = batch
            labels.append(y.cpu().numpy())
            res_dict = self.val_on_batch(epoch, batch_idx, batch, vis_flag)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        torch.cuda.empty_cache()
        # sap_score = _compute_sap(np.concatenate(ret['mu'], 0),
        #                          np.concatenate(labels, 0))
        sap_score = 0
        ret = {k: np.mean(v) if k not in ['val_images'] else v[0]
               for k, v in ret.items() if k not in ['mu']}
        ret["val_sap_score"] = sap_score
        return ret

    @torch.no_grad()
    def test_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        test_loss_meter = BasicMeter.get("test_loss").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(data_loader):
            mse, regularizer, loss = self.val_on_batch(batch_idx, batch, False)
            test_loss_meter.update(float(loss), 1)
        return {"test_loss": test_loss_meter.mean()}

    def get_state_dict(self):
        ret = {}
        ret["model"] = self.model.state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        if self.exp_dict["amp"] > 0:
            ret["amp"] = self.scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
        if self.exp_dict["amp"] > 0:
            self.scaler.load_state_dict(state_dict["amp"])

    def get_lr(self):
        ret = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            ret["current_lr_%d" % i] = float(param_group["lr"])
        return ret

    def is_end_of_training(self):
        lr = self.get_lr()["current_lr_0"]
        return lr <= (self.exp_dict["lr"] * self.exp_dict["min_lr_decay"])


def l1_loss(x, reconstruction):
    pix_mse = F.l1_loss(x, reconstruction, reduction="mean")
    pix_mse = pix_mse * (pix_mse != 0)  # masking to avoid nan
    return pix_mse
