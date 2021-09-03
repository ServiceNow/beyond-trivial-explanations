"""
Few-Shot Parallel: trains a model as a series of tasks computed in parallel on multiple GPUs

"""
import numpy as np
import os
import torch
import torch.nn.functional as F
from src.tools.meters import BasicMeter
# from trainers.few_shot_vanilla import FewShotVanillaTrainer
# from mixup_cifar10.train import mixup_criterion, mixup_data
from src.wrappers.base_wrapper import BaseWrapper
from src.models.mlp import MLP
import pylab

beta=0.0001

class Wrapper(BaseWrapper):
    """Trains a model on multiple GPUs"""

    def __init__(self, model, exp_dict, savedir):
        """ Constructor
        Args:
            model: architecture to train
            self.exp_dict: reference to dictionary with the global state of the application
        """
        super().__init__()
        self.model = model
        self.exp_dict = exp_dict 
        self.ngpu = self.exp_dict["ngpu"]
        self.savedir = savedir
        self.model.discriminator = MLP(self.model.output_size, 2)

        # Add optimizers here
        # self.optimizer = torch.optim.SGD(self.model.parameters(), 
        #                                     lr=self.exp_dict["lr"],
        #                                     momentum=0.9,
        #                                     nesterov=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.exp_dict["lr"])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode="min" if "loss" in self.exp_dict["target_loss"] else "max",
                                                                    patience=self.exp_dict["patience"])
        self.model.cuda()
        if self.ngpu > 1:
            self.parallel_model = torch.nn.DataParallel(self.model, device_ids=list(range(self.ngpu)))

    def save_img(self, x_orig, x_reco, idx=0):
        im_orig = x_orig.permute(1,2,0).data.cpu().numpy()
        im_reco = x_reco.permute(1,2,0).data.cpu().numpy()
        im = np.concatenate([im_orig, im_reco], 1)
        im = np.floor((im + 1) * 0.5 * 255)
        pylab.imsave(os.path.join(self.savedir, 'tmp_%d.jpg' %idx), im.astype('uint8'))

    def train_on_batch(self, batch_idx, batch, vis_flag):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        x.data = x.data * 2 - 1
        features, batch_recon, mu, logvar = self.model(x)
        features_recon = self.model.features(batch_recon)
        d1 = self.model.discriminator(features)
        d2 = self.model.discriminator(features_recon)

        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        MSE = F.mse_loss(batch_recon, x, reduction='mean')

        CLSS = F.cross_entropy(d1, y, weight=torch.tensor([1,0.32], device=MSE.device)) 
        # Rethink this 
        DIST = F.mse_loss(d2, d1, reduction='mean')
        mask = (y == 0).float() + 0.32 * (y == 1).float()
        DIST = (DIST * mask[:, None]).mean()
        pred = d1.argmax(1).data
        TP = ((pred == 0) * (y == 0)).float().sum()
        FN = ((pred == 1) * (y == 0)).float().sum()
        TPR = TP / (TP + FN + 1e-6)

        loss = KLD * beta + MSE + CLSS + DIST * beta 

        # Visualization        
        if vis_flag and batch_idx % 20 == 0:
            self.save_img(x[0], batch_recon[0])

        return float(KLD), float(MSE), float(CLSS), float(DIST), float(TPR), loss

    def val_on_batch(self, batch):
        x, y = batch
        x = x.cuda()
        y = y.cuda()
        x.data = x.data * 2 - 1
        features, batch_recon, mu, logvar = self.model(x)
        features_recon = self.model.features(batch_recon)
        d1 = self.model.discriminator(features)
        d2 = self.model.discriminator(features_recon)

        KLD = -0.5 * beta * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        MSE = F.mse_loss(batch_recon, x, reduction='mean')

        CLSS = beta * F.cross_entropy(d1, y, weight=torch.tensor([1,0.32], device=MSE.device)) 
        DIST = F.mse_loss(d2, d1, reduction='none')
        # Y = 0 is defect, Y = 1 is good. There are 3200 Y=0 and 10000 y=1
        mask = (y == 0).float() + 0.32 * (y == 1).float()
        DIST = (DIST * mask[:, None]).mean()
        pred = d1.argmax(1).data
        TP = ((pred == 0) * (y == 0)).float().sum()
        FN = ((pred == 1) * (y == 0)).float().sum()
        TPR = TP / (TP + FN + 1e-6)
        
        return KLD, MSE, CLSS, DIST, TPR 

    def predict_on_batch(self, x):
        return self.model(x.cuda()) 

    def train_on_loader(self, data_loader, vis_flag=False):
        """Iterate over the training set

        Args:
            data_loader: iterable training data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        global beta
        self.model.train()
        train_mse = BasicMeter.get("train_mse").reset()
        train_kld = BasicMeter.get("train_kld").reset()
        train_dist = BasicMeter.get("train_distillation_loss").reset()
        train_tpr = BasicMeter.get("train_tpr").reset()
        train_loss = BasicMeter.get("train_loss").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(data_loader):
            self.optimizer.zero_grad()
            KLD, MSE, CLSS, DIST, TPR, loss = self.train_on_batch(batch_idx, batch, vis_flag)
            if MSE < 0.01:
                beta = min(beta + 0.001, 1)
            train_mse.update(MSE, 1)
            train_kld.update(KLD, 1)
            train_dist.update(DIST, 1)
            train_tpr.update(TPR, 1)
            train_loss.update(float(loss), 1)
            loss.backward()
            self.optimizer.step()
        return {"train_loss": train_loss.mean(),
                "train_mse": train_mse.mean(),
                "train_kld": train_kld.mean(),
                "train_distillation_loss": train_dist.mean(),
                "beta": beta,
                "train_tpr": train_tpr.mean()}
        

    @torch.no_grad()
    def val_on_loader(self, data_loader, vis_flag=False):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.model.eval()
        val_mse = BasicMeter.get("val_mse").reset()
        val_kld = BasicMeter.get("val_kld").reset()
        val_dist = BasicMeter.get("val_distillation_loss").reset()
        val_tpr = BasicMeter.get("val_tpr").reset()
        val_loss_meter = BasicMeter.get("val_total_loss").reset()

        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(data_loader):
            KLD, MSE, CLSS, DIST, TPR = self.val_on_batch(batch)
            val_kld.update(float(KLD), 1)
            val_mse.update(float(MSE), 1)
            val_dist.update(float(DIST), 1)
            val_tpr.update(float(TPR), 1)
            val_loss_meter.update(float(KLD + MSE + DIST), 1)
        # loss = BasicMeter.get(self.exp_dict["target_loss"], recursive=True, force=False).mean()
        # self.scheduler.step(loss)  # update the learning rate monitor
        return {"val_mse": val_mse.mean(),
                "val_kl": val_kld.mean(),
                "val_distillation": val_dist.mean(),
                "val_loss": val_loss_meter.mean(),
                "val_tpr": val_tpr.mean()}

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
            loss = self.val_on_batch(batch)
            test_loss_meter.update(float(loss), 1)
        return {"test_loss": test_loss_meter.mean()}

    def get_state_dict(self):
        ret = {}
        ret["optimizer"] = self.optimizer.state_dict()
        ret["model"] = self.model.state_dict()
        ret["scheduler"] = self.scheduler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def get_lr(self):
        ret = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            ret["current_lr_%d" % i] = float(param_group["lr"])
        return ret

    def is_end_of_training(self):
        lr = self.get_lr()["current_lr_0"]
        return lr <= (self.exp_dict["lr"] * self.exp_dict["min_lr_decay"])