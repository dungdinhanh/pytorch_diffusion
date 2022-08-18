from pytorch_diffusion.diffusion_rnn import DiffusionRNN
from pytorch_diffusion.diffusion_rnn_isolate_iter import DiffusionRNN_IsolateIter
import os
import time

if __name__ == "__main__":
    import sys, tqdm
    name = sys.argv[1] if len(sys.argv)>1 else "cifar10"
    bs = int(sys.argv[2]) if len(sys.argv)>2 else 1
    nb = int(sys.argv[3]) if len(sys.argv)>3 else 1
    save_folder = str(sys.argv[4]) if len(sys.argv) > 4 else "results"
    rnn_state = str(sys.argv[5]) if len(sys.argv) > 5 else None

    os.makedirs(save_folder, exist_ok=True)

    diffusion = DiffusionRNN_IsolateIter.from_pretrained(name, log_folder=save_folder, state_path=rnn_state)

    diffusion.training(bs, nb)



