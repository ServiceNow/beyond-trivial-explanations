import numpy as np
import os
import torch
import torch.nn.functional as F
from src.wrappers.base_wrapper import BaseWrapper


from src.models import get_model
from tqdm import tqdm
from sklearn.metrics import f1_score, matthews_corrcoef

class Classifier(BaseWrapper):
    """Trains a model on multiple GPUs"""

    def __init__(self, exp_dict, savedir, datadir):
        """ Constructor
        Args:
            model: architecture to train
            self.exp_dict: reference to dictionary with the global state of the application
        """
        super().__init__()
        self.ngpu = exp_dict["ngpu"]
        self.exp_dict = exp_dict 
        self.devices = list(range(self.ngpu))
        self.savedir = savedir
        self.feat_extract = get_model(exp_dict["model"], exp_dict, datadir).cuda()
        self.classifier = torch.nn.Linear(self.feat_extract.output_size, exp_dict["n_classes"]).cuda()
        # self.discriminator_loss = DiscriminatorLoss(self.discriminator)
        self.optimizer = torch.optim.Adam([{'params': self.feat_extract.parameters(), 'lr': exp_dict['lr']},
                                            {'params': self.classifier.parameters(), 'lr': exp_dict['lr'] * exp_dict["finetune_lr"]}],
                                            betas=(0.9, 0.999),
                                            weight_decay=1e-4)
        
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
        im = im.permute(1,2,0)
        im = im * 255
        im = im.cpu().numpy().astype('uint8')
        return im

    def get_beta(self, epoch):
        if self.exp_dict["beta"] > 0:
            if self.exp_dict["beta_annealing"] == True:
                cycle_size = (self.exp_dict["max_epoch"] // 4)
                _epoch = epoch % cycle_size
                ratio = max(1e-4, min(1, _epoch / (cycle_size * 0.5)))
                beta = self.beta * ratio
            else:
                beta = self.beta
        else:
            beta = 0
        return beta

    def train_on_batch(self, epoch, batch_idx, batch):
        x, y = batch
        b, c, h, w = x.size()
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        b = x.size(0)
        self.optimizer.zero_grad()
        feats = torch.nn.DataParallel(self.feat_extract, list(range(self.ngpu)))(x)
        if self.ngpu > 1:
            logits = self.classifier(feats)
        else:
            logits = self.classifier(feats)
        if self.exp_dict["dataset_train"] in ['synbols', 'mnist', 'cifar']:
            # y = F.one_hot(y.long(), self.exp_dict["n_classes"])
            loss = F.cross_entropy(logits, y.long().view(-1))
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
        loss.backward()

        self.optimizer.step()

        return dict(train_loss=float(loss))

    def val_on_batch(self, epoch, batch_idx, batch, vis_flag):
        x, y = batch
        b = x.size(0)
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        feats = torch.nn.DataParallel(self.feat_extract, list(range(self.ngpu)))(x)
        if self.ngpu > 1:
            logits = self.classifier(feats)
        else:
            logits = self.classifier(feats)
        if self.exp_dict["dataset_train"] in ['synbols', 'mnist', 'cifar']:
            # y = F.one_hot(y.long(), self.exp_dict["n_classes"])
            y = y.view(-1)
            loss = F.cross_entropy(logits, y.long())
            preds = logits.argmax(-1)
            f1 = 0
            mcc = 0
        else:
            loss = F.binary_cross_entropy_with_logits(logits, y.float())
            preds = torch.round(torch.sigmoid(logits))
            f1 = f1_score(y.view(-1).data.cpu().numpy(), preds.view(-1).data.cpu().numpy())
            mcc = matthews_corrcoef(y.view(-1).data.cpu().numpy(), preds.view(-1).data.cpu().numpy()),
        acc = (preds == y).float().mean()
        return dict(val_loss=float(loss),
                   val_f1=f1,
                   mcc=mcc,
                   val_accuracy=float(acc))

    def predict_on_batch(self, x):
        return self.model(x.cuda()) 

    def train_on_loader(self, epoch, data_loader):
        """Iterate over the training set

        Args:
            data_loader: iterable training data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.train()
        self.n_data = len(data_loader.dataset)
        ret = {}
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            res_dict = {}
            res_dict = self.train_on_batch(epoch, batch_idx, batch)
            for k, v in res_dict.items():
                if k in ret:
                    ret[k].append(v)
                else:
                    ret[k] = [v]
        return {k: np.mean(v) for k,v in ret.items()}
        

    @torch.no_grad()
    def val_on_loader(self, epoch, data_loader, vis_flag=False):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.eval()
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
        ret = {k: np.mean(v) for k,v in ret.items() if k not in ['mu']}
        return ret

    @torch.no_grad()
    def test_on_loader(self, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.eval()
        test_loss_meter = BasicMeter.get("test_loss").reset()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        for batch_idx, batch in enumerate(data_loader):
            mse, regularizer, loss = self.val_on_batch(batch_idx, batch, False)
            test_loss_meter.update(float(loss), 1)
        return {"test_loss": test_loss_meter.mean()}

    def get_state_dict(self):
        ret = {}
        ret["feat_extract"] = self.feat_extract.state_dict()
        ret["classifier"] = self.classifier.state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.classifier.load_state_dict(state_dict["classifier"])
        self.feat_extract.load_state_dict(state_dict["feat_extract"])

    def get_lr(self):
        ret = {}
        for i, param_group in enumerate(self.optimizer.param_groups):
            ret["current_lr_%d" % i] = float(param_group["lr"])
        return ret

    def is_end_of_training(self):
        lr = self.get_lr()["current_lr_0"]
        return lr <= (self.exp_dict["lr"] * self.exp_dict["min_lr_decay"])

def l1_loss(x, reconstruction):
    pix_mse = F.l1_loss(x, reconstruction, reduction="sum")
    pix_mse = pix_mse * (pix_mse != 0) / x.size(0) # masking to avoid nan
    return pix_mse