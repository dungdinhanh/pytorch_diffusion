from pytorch_diffusion.diffusion_rnn import DiffusionRNN
import os
import time

if __name__ == "__main__":
    import sys, tqdm
    name = sys.argv[1] if len(sys.argv)>1 else "cifar10"
    bs = int(sys.argv[2]) if len(sys.argv)>2 else 1
    nb = int(sys.argv[3]) if len(sys.argv)>3 else 1
    save_folder = str(sys.argv[4]) if len(sys.argv) > 4 else "results"

    os.makedirs(save_folder, exist_ok=True)

    diffusion = DiffusionRNN.from_pretrained(name, train=False, state_path="../drive/MyDrive/usydphd/cifar10_25_1000/iter999.pth")
    for ib in tqdm.tqdm(range(nb), desc="Batch"):
        start_time = time.time()
        x = diffusion.inference(bs)
        idx = ib*bs
        exc_time = time.time() - start_time
        print("%f iter/s"%(1/exc_time))

        image_path = os.path.join(save_folder, name+"/{:06}.png")

        diffusion.save(x, image_path, start_idx=idx)
