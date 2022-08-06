from pytorch_diffusion.diffusion import Diffusion
import os

if __name__ == "__main__":
    import sys, tqdm
    name = sys.argv[1] if len(sys.argv)>1 else "cifar10"
    bs = int(sys.argv[2]) if len(sys.argv)>2 else 1
    nb = int(sys.argv[3]) if len(sys.argv)>3 else 1
    save_folder = str(sys.argv[4]) if len(sys.argv) > 4 else "results"

    os.makedirs(save_folder, exist_ok=True)

    diffusion = Diffusion.from_pretrained(name)
    for ib in tqdm.tqdm(range(nb), desc="Batch"):
        x = diffusion.denoise(bs, progress_bar=tqdm.tqdm)
        idx = ib*bs

        image_path = os.path.join(save_folder, name+"/{:06}.png")

        diffusion.save(x, image_path, start_idx=idx)