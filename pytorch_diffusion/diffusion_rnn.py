import numpy as np
import torch
from pytorch_diffusion.model import Model, ModelExtract, ModelRNN
from pytorch_diffusion.diffusion import *
from datasets.data_helper import *
from pytorch_diffusion.ckpt_util import get_ckpt_path
import torchvision.datasets as datasets
import os
import torch.optim as optim
import time


class DiffusionRNN(Diffusion):
    def __init__(self, diffusion_config, model_config, device=None, train=True,
                 lr=0.01, weight_decay=1e-4, data_loader=None):
        super(DiffusionRNN, self).__init__(diffusion_config=diffusion_config,
                                           model_config=model_config, device=device, extract_version=True)

        emb_res, emb_channel = self.model.emb_res, self.model.emb_channel
        self.model_rnn = ModelRNN(emb_res, emb_channel)
        if train:
            self.model_rnn.train()
        else:
            self.model_rnn.eval()

        if train:
            self.data_loader, _ = get_cifar_loader(train=train, test=False)
        else:
            _, self.data_loader = get_cifar_loader(train=train, test=True)

        self.optimizer = optim.SGD(self.model_rnn.parameters(), lr=lr, weight_decay=weight_decay) # use SGD first
        # will change later to see performance
        self.loss_function = torch.nn.MSELoss(reduction='mean')

    def inference(self, n):
        self.model_rnn.eval()
        x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution)
        x, hs, temb = self.model.forward_down_mid(x_0, 0)

        for i in range(self.num_timesteps):
            downs = False
            ups=False
            if i == 0:
                downs = True
                h = None
            if i == (self.num_timesteps - 1):
                ups = True
            h, c, x_prime, _ = self.model_rnn(x, h, downs, ups)
            x = x_prime

        x_final = self.model.forward_up(x, hs, temb)

        return x + x_final







    def denoise(self, n, n_steps=None, x=None, curr_step=None,
                progress_bar=lambda i, total=None: i,
                callback=lambda x, i, x0=None: None):
        with torch.no_grad():
            if curr_step is None:
                curr_step = self.num_timesteps

            assert curr_step > 0, curr_step

            if n_steps is None or curr_step-n_steps < 0:
                n_steps = curr_step

            if x is None:
                assert curr_step == self.num_timesteps, curr_step
                # start the chain with x_T from normal distribution
                x = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution)
                x = x.to(self.device)

            for i in progress_bar(reversed(range(curr_step-n_steps, curr_step)), total=n_steps):
                t = (torch.ones(n)*i).to(self.device)
                x, x0 = denoising_step(x,
                                       t=t,
                                       model=self.model,
                                       logvar=self.logvar,
                                       sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                                       sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                                       posterior_mean_coef1=self.posterior_mean_coef1,
                                       posterior_mean_coef2=self.posterior_mean_coef2,
                                       return_pred_xstart=True)
                callback(x, i, x0=x0)
            return x


    def diffuse(self, n, n_steps=None, x=None, curr_step=None,
                progress_bar=lambda i, total=None: i,
                callback=lambda x, i: None):
        with torch.no_grad():
            if curr_step is None:
                curr_step = 0

            assert curr_step < self.num_timesteps, curr_step

            if n_steps is None or curr_step+n_steps > self.num_timesteps:
                n_steps = self.num_timesteps-curr_step

            assert x is not None

            for i in progress_bar(range(curr_step, curr_step+n_steps), total=n_steps):
                t = (torch.ones(n)*i).to(self.device)
                x = diffusion_step(x,
                                   t=t,
                                   sqrt_alphas=self.sqrt_alphas,
                                   sqrt_one_minus_alphas=self.sqrt_one_minus_alphas)
                callback(x, i+1)

            return x



if __name__ == "__main__":
    import sys, tqdm
    name = sys.argv[1] if len(sys.argv)>1 else "cifar10"
    bs = int(sys.argv[2]) if len(sys.argv)>2 else 1
    nb = int(sys.argv[3]) if len(sys.argv)>3 else 1
    save_folder = str(sys.argv[4]) if len(sys.argv) > 4 else "results"

    os.makedirs(save_folder, exist_ok=True)

    diffusion = DiffusionRNN.from_pretrained(name)
    for ib in tqdm.tqdm(range(nb), desc="Batch"):
        start_time = time.time()
        x = diffusion.inference(bs)
        idx = ib*bs
        exc_time = time.time() - start_time
        print("%f iter/s"%(1/exc_time))

        image_path = os.path.join(save_folder, name+"/{:06}.png")

        diffusion.save(x, image_path, start_idx=idx)
