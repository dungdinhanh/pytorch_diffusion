from pytorch_diffusion.diffusion import *
import os
import torch.optim as optim
import time
from   datasets.data_helper import *
import random
from torch.utils.tensorboard import SummaryWriter


def denoising_step_rnn(model_sc_output, x, t,*,
                   logvar,
                   sqrt_recip_alphas_cumprod,
                   sqrt_recipm1_alphas_cumprod,
                   posterior_mean_coef1,
                   posterior_mean_coef2,
                   return_pred_xstart=False):
    """
    Sample from p(x_{t-1} | x_t)
    """
    # instead of using eq. (11) directly, follow original implementation which,
    # equivalently, predicts x_0 and uses it to compute mean of the posterior
    # 1. predict eps via model
    model_output = model_sc_output
    # 2. predict clipped x_0
    # (follows from x_t=sqrt_alpha_cumprod*x_0 + sqrt_one_minus_alpha*eps)
    pred_xstart = (extract(sqrt_recip_alphas_cumprod, t, x.shape)*x -
                   extract(sqrt_recipm1_alphas_cumprod, t, x.shape)*model_output)
    pred_xstart = torch.clamp(pred_xstart, -1, 1)
    # 3. compute mean of q(x_{t-1} | x_t, x_0) (eq. (6))
    mean = (extract(posterior_mean_coef1, t, x.shape)*pred_xstart +
            extract(posterior_mean_coef2, t, x.shape)*x)

    logvar = extract(logvar, t, x.shape)

    # sample - return mean for t==0
    noise = torch.randn_like(x)
    mask = 1-(t==0).float()
    mask = mask.reshape((x.shape[0],)+(1,)*(len(x.shape)-1))
    sample = mean + mask*torch.exp(0.5*logvar)*noise
    sample = sample.float()
    if return_pred_xstart:
        return sample, mean, pred_xstart
    return sample, mean


def avg_accumulate(h_emb_cal, n, h_emb):
    return (h_emb_cal + h_emb)/(n+1), n+1


def compare_state_dict(model1: torch.nn.Module, model2: torch.nn.Module):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    count = 0
    for key in state_dict1.keys():
        print(state_dict2[key] - state_dict1[key])
        if count > 5:
            break
    pass


class DiffusionRNN(Diffusion):
    def __init__(self, diffusion_config, model_config, device=None, train=True,
                 lr=0.01, weight_decay=1e-4, data_loader=None, log_folder="./runs"):
        super(DiffusionRNN, self).__init__(diffusion_config=diffusion_config,
                                           model_config=model_config, device=device, extract_version=True)

        emb_res, emb_channel = self.model.emb_res, self.model.emb_channel
        self.model_rnn = ModelRNN(emb_res, emb_channel)
        self.test_model = ModelRNN(emb_res, emb_channel)
        self.model_rnn.to(self.device)
        self.test_model.to(self.device)
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
        self.tensorboard_writer = SummaryWriter(os.path.join(log_folder, "log"))
        self.log_folder = log_folder
        self.folder_path = os.path.join(self.log_folder, "models")
        self.start_iter = None
        os.makedirs(self.folder_path, exist_ok=True)

    def training(self, n, number_of_iters=10000):
        self.model_rnn.train()
        self.model.eval()
        if self.start_iter is None:
            self.start_iter = 0
        x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution).to(self.device)
        for i in range(self.start_iter, number_of_iters, 1):

            # rand_number_timesteps = random.randint(5, self.num_timesteps-1)
            rand_number_timesteps = random.randint(5 , 20)
            start_step = random.randint(1, self.num_timesteps - rand_number_timesteps - 1)
            stop_step = start_step + rand_number_timesteps - 1

            print("iter %d: start at %d and stop at %d, number of step: %d" % (i, start_step, stop_step, rand_number_timesteps))

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
                    with torch.no_grad():
                        if j == 1:
                            down_sample = True
                        else:
                            down_sample = False
                        up_sample = True
                        h_rnn, c_rnn, x_prime, out_x_prime = self.model_rnn(h_emb, hx, down_sample, up_sample)
                        hx = (h_rnn, c_rnn)
                        h_emb = x_prime
                else:

                    if j == start_step:
                        down_sample=True
                        h_emb = h
                    else:
                        down_sample=False
                    up_sample=True
                    h_rnn, c_rnn, x_prime, out_x_prime = self.model_rnn(h_emb, hx, down_sample, up_sample)
                    hx = (h_rnn,c_rnn)
                    h_emb = x_prime

                    # RNN output
                    model_rnn_output = self.model.forward_up(out_x_prime, hs, temb)

                    sample_rnn, mean_rnn, xpred_rnn = denoising_step_rnn(
                                                                    model_rnn_output,
                                                                    x=x,
                                                                    t=t,
                                                                    logvar=self.logvar,
                                                                    sqrt_recip_alphas_cumprod=
                                                                    self.sqrt_recip_alphas_cumprod,
                                                                    sqrt_recipm1_alphas_cumprod=
                                                                    self.sqrt_recipm1_alphas_cumprod,
                                                                    posterior_mean_coef1=self.posterior_mean_coef1,
                                                                    posterior_mean_coef2=self.posterior_mean_coef2,
                                                                    return_pred_xstart=True)

                    # RNN accumulate
                    if h_emb_accumulate is None:
                        h_emb_accumulate = torch.zeros_like(h_emb).to(self.device)
                        count_accumulate = 0

                    h_emb_accumulate, count_accumulate = avg_accumulate(h_emb_accumulate, count_accumulate, h_emb)
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

                    while len(hs) != 0:
                        hs.pop()

                    loss_iter = self.loss_function(mean_rnn, mean)
                    loss_accumulate = self.loss_function(mean_accumulate, mean)
                x = sample
            self.optimizer.zero_grad()
            final_loss = loss_iter + loss_accumulate
            final_loss.backward()
            self.optimizer.step()

            print(final_loss.item())
            self.tensorboard_writer.add_scalar("Loss/train", final_loss.item(), i)
            self.tensorboard_writer.add_scalar("Loss/train_loss_iter", loss_iter.item(), i)
            self.tensorboard_writer.add_scalar("Loss/train_loss_accumulate", loss_accumulate.item(), i)
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
