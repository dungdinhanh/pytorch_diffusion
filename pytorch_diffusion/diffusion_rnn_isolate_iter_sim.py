from pytorch_diffusion.diffusion import *
import os
import torch.optim as optim
import time
from   datasets.data_helper import *
import random
from torch.utils.tensorboard import SummaryWriter
from pytorch_diffusion.diffusion_rnn import *
from pytorch_diffusion.model import ModelRNNXE


class DiffusionRNN_IsolateIterSim(DiffusionRNN):
    def __init__(self, diffusion_config, model_config, device=None, train=True,
                 lr=0.001, weight_decay=1e-4, data_loader=None, log_folder="./runs"):
        super(DiffusionRNN_IsolateIterSim, self).__init__(diffusion_config, model_config, device, train, lr,
                                                       weight_decay, data_loader, log_folder)
        emb_res, emb_channel = self.model.emb_res, self.model.emb_channel
        self.model_rnn = ModelRNNXE(emb_res, emb_channel)
        self.model_rnn.to(self.device)
        if train:
            self.model_rnn.train()
        else:
            self.model_rnn.eval()

        self.optimizer = optim.SGD(self.model_rnn.parameters(), lr=lr, weight_decay=weight_decay)  # use SGD first

    def training(self, n, number_of_iters=10000):
        self.model_rnn.train()
        self.model.eval()
        if self.start_iter is None:
            self.start_iter = 0

        x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution).to(self.device)

        for i in range(self.start_iter, number_of_iters, 1):

            # rand_number_timesteps = random.randint(5, self.num_timesteps-1)
            rand_number_timesteps = 10
            # start_step = random.randint(1, self.num_timesteps - rand_number_timesteps)
            start_step = 990
            stop_step = start_step + rand_number_timesteps - 1

            print("iter %d: start at %d and stop at %d, number of step: %d" % (i, start_step, stop_step, rand_number_timesteps))
            # x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution).to(self.device)
            x = x_0

            t = (torch.ones(n) * 0).to(self.device)
            h_emb, hs_0, temb_0 = self.model.forward_down_mid(x, t)
            model_sc_output = self.model.forward_up(h_emb, hs_0, temb_0)
            sample, mean, xpred = denoising_step_rnn(
                model_sc_output=model_sc_output,
                x=x,
                t=t,
                logvar=self.logvar,
                sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                posterior_mean_coef1=self.posterior_mean_coef1,
                posterior_mean_coef2=self.posterior_mean_coef2,
                return_pred_xstart=True)

            x = sample
            start = 1
            hx = None
            h_emb_accumulate = None
            count_accumulate = 0
            loss_iter = 0.0
            loss_accumulate = 0.0
            # for name, param in self.model_rnn.named_parameters():
            #     print(name)
            sample_rnn = None

            for j in range(start, stop_step, 1):
                t = (torch.ones(n) * j).to(self.device)
                with torch.no_grad():
                    h, hs, temb = self.model.forward_down_mid(x, t)

                    model_sc_output = self.model.forward_up(h, hs, temb)

                    sample, mean, xpred = denoising_step_rnn(
                        model_sc_output=model_sc_output,
                        x=x,
                        t=t,
                        logvar=self.logvar,
                        sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                        sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                        posterior_mean_coef1=self.posterior_mean_coef1,
                        posterior_mean_coef2=self.posterior_mean_coef2,
                        return_pred_xstart=True)

                if j < start_step:
                    if j == 1:
                        down_sample = True
                    else:
                        down_sample = False
                    up_sample = False
                    h_rnn, c_rnn, x_prime, out_x_prime = self.model_rnn(h_emb, hx, down_sample, up_sample)
                    hx = (h_rnn, c_rnn)
                    h_emb = x_prime
                else:

                    up_sample=True
                    down_sample=False
                    h_rnn, c_rnn, x_prime, out_x_prime = self.model_rnn(h_emb, hx, down_sample, up_sample)
                    hx = (h_rnn,c_rnn)
                    h_emb = x_prime
                    loss_iter = self.loss_function(out_x_prime, h)
                x = sample
            self.optimizer.zero_grad()
            final_loss = loss_iter
            final_loss.backward()
            self.optimizer.step()

            print(final_loss.item())
            self.tensorboard_writer.add_scalar("Loss/train", final_loss.item(), i)
            self.tensorboard_writer.add_scalar("Loss/train_loss_iter", loss_iter.item(), i)
            if i % 10 == 0:
                self.tensorboard_writer.flush()
            if i % 100 == 0 or i == number_of_iters - 1:
                state = {
                    'iter': i,
                    'optimizer': self.optimizer.state_dict(),
                    'state_dict': self.model_rnn.state_dict()
                }

                model_path = os.path.join(self.folder_path, "iter%d.pth"%i)
                torch.save(state, model_path)
        self.tensorboard_writer.flush()
        self.tensorboard_writer.close()


class DiffusionRNN_IsolateIterSim_Rand(DiffusionRNN_IsolateIterSim):
    def __init__(self, diffusion_config, model_config, device=None, train=True,
                 lr=0.001, weight_decay=1e-4, data_loader=None, log_folder="./runs"):
        super(DiffusionRNN_IsolateIterSim, self).__init__(diffusion_config, model_config, device, train, lr,
                                                          weight_decay, data_loader, log_folder)

    def training(self, n, number_of_iters=10000):
        self.model_rnn.train()
        self.model.eval()
        if self.start_iter is None:
            self.start_iter = 0

        x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution).to(self.device)

        for i in range(self.start_iter, number_of_iters, 1):

            # rand_number_timesteps = random.randint(5, self.num_timesteps-1)
            rand_number_timesteps = 20
            start_step = random.randint(1, self.num_timesteps - rand_number_timesteps)
            stop_step = start_step + rand_number_timesteps - 1

            print("iter %d: start at %d and stop at %d, number of step: %d" % (
            i, start_step, stop_step, rand_number_timesteps))
            # x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution).to(self.device)
            x = x_0

            t = (torch.ones(n) * 0).to(self.device)
            h_emb, hs_0, temb_0 = self.model.forward_down_mid(x, t)
            model_sc_output = self.model.forward_up(h_emb, hs_0, temb_0)
            sample, mean, xpred = denoising_step_rnn(
                model_sc_output=model_sc_output,
                x=x,
                t=t,
                logvar=self.logvar,
                sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                posterior_mean_coef1=self.posterior_mean_coef1,
                posterior_mean_coef2=self.posterior_mean_coef2,
                return_pred_xstart=True)

            x = sample
            start = 1
            hx = None
            h_emb_accumulate = None
            count_accumulate = 0
            loss_iter = 0.0
            loss_accumulate = 0.0
            # for name, param in self.model_rnn.named_parameters():
            #     print(name)
            sample_rnn = None

            for j in range(start, stop_step, 1):
                t = (torch.ones(n) * j).to(self.device)
                with torch.no_grad():
                    h, hs, temb = self.model.forward_down_mid(x, t)

                    model_sc_output = self.model.forward_up(h, hs, temb)

                    sample, mean, xpred = denoising_step_rnn(
                        model_sc_output=model_sc_output,
                        x=x,
                        t=t,
                        logvar=self.logvar,
                        sqrt_recip_alphas_cumprod=self.sqrt_recip_alphas_cumprod,
                        sqrt_recipm1_alphas_cumprod=self.sqrt_recipm1_alphas_cumprod,
                        posterior_mean_coef1=self.posterior_mean_coef1,
                        posterior_mean_coef2=self.posterior_mean_coef2,
                        return_pred_xstart=True)

                if j < start_step:
                    if j == 1:
                        down_sample = True
                    else:
                        down_sample = False
                    up_sample = False
                    h_rnn, c_rnn, x_prime, out_x_prime = self.model_rnn(h_emb, hx, down_sample, up_sample)
                    hx = (h_rnn, c_rnn)
                    h_emb = x_prime
                else:

                    up_sample = True
                    down_sample = False
                    h_rnn, c_rnn, x_prime, out_x_prime = self.model_rnn(h_emb, hx, down_sample, up_sample)
                    hx = (h_rnn, c_rnn)
                    h_emb = x_prime
                    loss_iter = self.loss_function(out_x_prime, h)
                x = sample
            self.optimizer.zero_grad()
            final_loss = loss_iter
            final_loss.backward()
            self.optimizer.step()

            print(final_loss.item())
            self.tensorboard_writer.add_scalar("Loss/train", final_loss.item(), i)
            self.tensorboard_writer.add_scalar("Loss/train_loss_iter", loss_iter.item(), i)
            if i % 10 == 0:
                self.tensorboard_writer.flush()
            if i % 100 == 0 or i == number_of_iters - 1:
                state = {
                    'iter': i,
                    'optimizer': self.optimizer.state_dict(),
                    'state_dict': self.model_rnn.state_dict()
                }

                model_path = os.path.join(self.folder_path, "iter%d.pth" % i)
                torch.save(state, model_path)
        self.tensorboard_writer.flush()
        self.tensorboard_writer.close()
