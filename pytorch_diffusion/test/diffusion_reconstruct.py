from pytorch_diffusion.diffusion import *
import os
import torch.optim as optim
import time
from   datasets.data_helper import *
import random
from torch.utils.tensorboard import SummaryWriter
from pytorch_diffusion.diffusion_rnn import *
from pytorch_diffusion.test.model_test import *
from torchvision.utils import save_image
from pytorch_fid.fid_score import calculate_fid_given_paths


class DiffusionReconstruct(Diffusion):
    def __init__(self, diffusion_config, model_config, device=None, train=True,
                 lr=0.001, weight_decay=1e-4, data_loader=None, log_folder="./runs", bs=64, load_plain_image=False):
        super(DiffusionReconstruct, self).__init__(diffusion_config, model_config, device, True)
        self.decoder_model = ModelReconstruct(**self.model_config, emb_res=self.model.emb_res,
                                              block_in=self.model.emb_channel)
        self.decoder_model.to(self.device)
        self.train = train
        if train:
            self.optimizer = optim.SGD(self.decoder_model.parameters(), lr=lr, weight_decay=weight_decay)

        self.loss_function = torch.nn.MSELoss(reduction='mean')
        self.tensorboard_writer = SummaryWriter(os.path.join(log_folder, "log"))
        self.log_folder = log_folder
        self.folder_path = os.path.join(self.log_folder, "models")
        self.bs = bs
        os.makedirs(self.folder_path, exist_ok=True)

        if train:
            self.data_loader, _ = get_cifar_loader(train=train, test=False, batch_size=bs)
        else:
            _, self.data_loader = get_cifar_loader(train=train, test = True, batch_size=bs)

        if load_plain_image:
            if train:
                cifar_traintest = datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
                self.train_plain_folder = os.path.join("./data", "cifar10_train_pi")
                os.makedirs(self.train_plain_folder, exist_ok=True)
                for i in range(len(cifar_traintest)):
                    img, target = cifar_traintest[i]
                    img.save(os.path.join(self.train_plain_folder, "IMG{:06}.png".format(i)))
            else:
                cifar_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
                self.test_plain_folder = os.path.join("./data", "cifar10_test_pi")
                os.makedirs(self.test_plain_folder, exist_ok=True)
                for i in range(len(cifar_testset)):
                    img, target = cifar_testset[i]
                    img.save(os.path.join(self.test_plain_folder, "IMG{:06}.png".format(i)))


    def training(self, n, number_of_iters=100):
        self.model.eval()
        self.decoder_model.train()
        i_iter = 0
        for i in range(number_of_iters):
            for j, batch in enumerate(self.data_loader, 0):
            # x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution).to(self.device)
                x, _ = batch
                x = x.to(self.device)
                t = (torch.ones(x.shape[0]) * 1000).to(self.device)
                h_emb, hs_0, temb_0 = self.model.forward_down_mid(x, t)
                x_prime = self.decoder_model(h_emb, temb_0)
                loss_iter = self.loss_function(x_prime, x)

                self.optimizer.zero_grad()
                final_loss = loss_iter
                final_loss.backward()
                self.optimizer.step()

                self.tensorboard_writer.add_scalar("Loss/train_loss_iter", loss_iter.item(), i_iter)

                if i_iter % 100  == 0 or i_iter == number_of_iters - 1:
                    print("%d epoch - %d iter: Loss %f" % (i, j, final_loss.item()))
                if i_iter % 400 == 0 or i_iter == number_of_iters - 1:
                    state = {
                        'iter': i,
                        'optimizer': self.optimizer.state_dict(),
                        'state_dict': self.decoder_model.state_dict()
                    }
                    model_path = os.path.join(self.folder_path, "epoch%diter%d.pth" % (i, j))
                    torch.save(state, model_path)

                if i_iter % 10 == 0:
                    self.tensorboard_writer.flush()
                i_iter += 1

        self.tensorboard_writer.flush()
        self.tensorboard_writer.close()

    def inference(self):
        self.decoder_model.eval()
        self.model.eval()
        self.inference_path = os.path.join(self.log_folder, "inference")
        os.makedirs(self.inference_path, exist_ok=True)
        count_image = 0
        for j, batch in enumerate(self.data_loader, 0):
            # x_0 = torch.randn(n, self.model.in_channels, self.model.resolution, self.model.resolution).to(self.device)
            x, _ = batch
            x = x.to(self.device)
            t = (torch.ones(x.shape[0]) * 1000).to(self.device)
            h_emb, hs_0, temb_0 = self.model.forward_down_mid(x, t)
            x_prime = self.decoder_model(h_emb, temb_0)
            for i in range(x_prime.shape[0]):
                save_image(x_prime[i], os.path.join(self.inference_path, "im%d.png"%(count_image)))
                count_image += 1

        if self.train:
            target_folder = self.train_plain_folder
        else:
            target_folder = self.test_plain_folder

        calculate_fid_given_paths([self.inference_path, target_folder], self.bs, self.device, num_workers=4, dims=2048)

    @classmethod
    def from_pretrained(cls, name, train=True ,device=None, log_folder="./runs/", state_path=None, bs=64, load_plain_image=False):
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
        diffusion = cls(diffusion_config, model_config_map[basename], device, train, log_folder=log_folder, bs=bs,
                        load_plain_image=load_plain_image)

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
            diffusion.decoder_model.load_state_dict(state['state_dict'])
            if train:
                diffusion.optimizer.load_state_dict(state['optimizer'])
                diffusion.start_iter = state['iter']
            diffusion.decoder_model.to(diffusion.device)
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
