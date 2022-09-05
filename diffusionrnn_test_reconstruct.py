from pytorch_diffusion.test.diffusion_reconstruct import *
import os

if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv)>1 else "cifar10"
    bs = int(sys.argv[2]) if len(sys.argv)>2 else 1
    nb = int(sys.argv[3]) if len(sys.argv)>3 else 1
    save_folder = str(sys.argv[4]) if len(sys.argv) > 4 else "results"
    rnn_state = str(sys.argv[5]) if len(sys.argv) > 5 else None
    plain_image = str(sys.argv[6]) if len(sys.argv) > 6 else False

    os.makedirs(save_folder, exist_ok=True)

    diffusion = DiffusionReconstruct.from_pretrained(name, train=False, load_plain_image=True ,log_folder=save_folder,
                                                     state_path=rnn_state, bs=bs)

    diffusion.inference()



