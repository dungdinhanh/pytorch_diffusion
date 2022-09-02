from pytorch_diffusion.diffusion import *
import os
import torch.optim as optim
import time
from   datasets.data_helper import *
import random
from torch.utils.tensorboard import SummaryWriter
from pytorch_diffusion.diffusion_rnn import *
from pytorch_diffusion.test.model_test import *


class DiffusionReconstruct(Diffusion):
    def __init__(self, diffusion_config, model_config, device=None, train=True,
                 lr=0.001, weight_decay=1e-4, data_loader=None, log_folder="./runs"):
        super(DiffusionReconstruct, self).__init__(diffusion_config, model_config, device, train)
        self.decoder_model = ModelReconstruct(**self.model_config)
        self.decoder_model.to(self.device)

        self.optimizer = optim.SGD(self.decoder_model.parameters(), lr=lr, weight_decay=weight_decay)

        self.loss_function = torch.nn.MSELoss(reduction='mean')
        self.tensorboard_writer = SummaryWriter(os.path.join(log_folder, "log"))
        self.log_folder = log_folder
        self.folder_path = os.path.join(self.log_folder, "models")
        os.makedirs(self.folder_path, exist_ok=True)

        if train:
            self.data_loader, _ = get_cifar_loader(train=train, test=False)
        else:
            _, self.data_loader = get_cifar_loader(train=train, test = True)

    def training(self, n, number_of_iters=10000):
        self.model.eval()
        self.decoder_model.train()
        dataloader_iter = iter(self.data_loader)
        for i in range(number_of_iters):
            # x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution).to(self.device)
            x, _ = next(dataloader_iter)
            t = (torch.ones(n) * 1000).to(self.device)
            h_emb, hs_0, temb_0 = self.model.forward_down_mid(x, t)
            x_prime = self.decoder_model(h_emb)
            loss_iter = self.loss_function(x_prime, x)

            self.optimizer.zero_grad()
            final_loss = loss_iter
            final_loss.backward()
            self.optimizer.step()
            print("%d iter: Loss %f"%(i, final_loss.item()))
            self.tensorboard_writer.add_scalar("Loss/train_loss_iter", loss_iter.item(), i)
            if i % 100 == 0 or i == number_of_iters - 1:
                state = {
                    'iter': i,
                    'optimizer': self.optimizer.state_dict(),
                    'state_dict': self.decoder_model.state_dict()
                }
                model_path = os.path.join(self.folder_path, "iter%d.pth" % i)
                torch.save(state, model_path)
            if i % 10 == 0:
                self.tensorboard_writer.flush()

        self.tensorboard_writer.flush()
        self.tensorboard_writer.close()

    def inference(self, n):
        self.model_rnn.eval()
        self.model.eval()
        x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution).to(self.device)
        t = (torch.ones(n) * 0).to(self.device)
        h_0, hs_0, temb_0 = self.model.forward_down_mid(x_0, t)
        h = h_0
        h_emb_accumulate = None
        count_accumulate = 0
        for i in range(self.num_timesteps):
            downs = False
            ups=False
            if i == 0:
                downs = True
                hx = None
            if i == (self.num_timesteps - 1):
                ups = True
            h_rnn, c_rnn, h_prime_sample, h_prime = self.model_rnn(h, hx, downs, ups)
            hx = (h_rnn, c_rnn)
            h = h_prime_sample

            # RNN accumulate
            if h_emb_accumulate is None:
                h_emb_accumulate = torch.zeros_like(h).to(self.device)
                count_accumulate = 0

            h_emb_accumulate, count_accumulate = avg_accumulate(h_emb_accumulate, count_accumulate, h)
        c_h_emb_accumulate = c_rnn + torch.rand_like(h_emb_accumulate) * h_emb_accumulate
        accumulate_x_prime = self.model_rnn.forward_dec(c_h_emb_accumulate)
        model_accumulate_output = self.model.forward_up(accumulate_x_prime, hs_0, temb_0)
        sample_accumulate, mean_accumulate, xpred_accumulate = denoising_step_rnn(
            model_accumulate_output,
            x=x_0,
            t=t,
            logvar=self.logvar,
            sqrt_recip_alphas_cumprod=
            self.sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod=
            self.sqrt_recipm1_alphas_cumprod,
            posterior_mean_coef1=self.posterior_mean_coef1,
            posterior_mean_coef2=self.posterior_mean_coef2,
            return_pred_xstart=True)
        # must change the return later depends on the design of training
        return sample_accumulate

    @classmethod
    def from_pretrained(cls, name, train=True ,device=None, log_folder="./runs/", state_path=None):
        cifar10_cfg = {
            "resolution": 32,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1, 2, 2, 2),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.1,
        }
        lsun_cfg = {
            "resolution": 256,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": (1, 1, 2, 2, 4, 4),
            "num_res_blocks": 2,
            "attn_resolutions": (16,),
            "dropout": 0.0,
        }

        model_config_map = {
            "cifar10": cifar10_cfg,
            "lsun_bedroom": lsun_cfg,
            "lsun_cat": lsun_cfg,
            "lsun_church": lsun_cfg,
        }

        diffusion_config = {
            "beta_schedule": "linear",
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "num_diffusion_timesteps": 1000,
        }
        model_var_type_map = {
            "cifar10": "fixedlarge",
            "lsun_bedroom": "fixedsmall",
            "lsun_cat": "fixedsmall",
            "lsun_church": "fixedsmall",
        }
        ema = name.startswith("ema_")
        basename = name[len("ema_"):] if ema else name
        diffusion_config["model_var_type"] = model_var_type_map[basename]

        print("Instantiating")
        # later will add rnn configs for training
        diffusion = cls(diffusion_config, model_config_map[basename], device, train, log_folder=log_folder)

        ckpt = get_ckpt_path(name)
        print("Loading checkpoint {}".format(ckpt))
        diffusion.model.load_state_dict(torch.load(ckpt, map_location=diffusion.device))
        diffusion.model.to(diffusion.device)
        diffusion.model.eval()
        print("Moved model to {}".format(diffusion.device))

        print("Loading checkpoint for model rnn {}".format(state_path))
        if  state_path is None:
            print("No state information to load into rnn model")
            print("Random initialization")
        elif not os.path.isfile(state_path):
            print("%s can not be found"%state_path)
        else:
            state = torch.load(state_path, map_location=diffusion.device)
            diffusion.model_rnn.load_state_dict(state['state_dict'])
            if train:
                diffusion.optimizer.load_state_dict(state['optimizer'])
                diffusion.start_iter = state['iter']
            diffusion.model_rnn.to(diffusion.device)
        return diffusion

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
        idx = ib * bs
        exc_time = time.time() - start_time
        print("%f iter/s"%(1/exc_time))

        image_path = os.path.join(save_folder, name+"/{:06}.png")

        diffusion.save(x, image_path, start_idx=idx)
