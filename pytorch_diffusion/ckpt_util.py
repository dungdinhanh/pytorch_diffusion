import os, hashlib
import requests
from tqdm import tqdm

URL_MAP = {
    "cifar10": "https://heibox.uni-heidelberg.de/f/869980b53bf5416c8a28/?dl=1",
    "lsun_bedroom": "https://heibox.uni-heidelberg.de/f/f179d4f21ebc4d43bbfe/?dl=1",
    "lsun_cat": "https://heibox.uni-heidelberg.de/f/fac870bd988348eab88e/?dl=1",
    "lsun_church": "https://heibox.uni-heidelberg.de/f/2711a6f712e34b06b9d8/?dl=1",
}
CKPT_MAP = {
    "cifar10": "diffusion_cifar10_model/model-790000.ckpt",
    "lsun_bedroom": "diffusion_lsun_bedroom_model/model-2388000.ckpt",
    "lsun_cat": "diffusion_lsun_cat_model/model-1761000.ckpt",
    "lsun_church": "diffusion_lsun_church_model/model-4432000.ckpt",
}
MD5_MAP = {
    "cifar10": "82ed3067fd1002f5cf4c339fb80c4669",
    "lsun_bedroom": "f70280ac0e08b8e696f42cb8e948ff1c",
    "lsun_cat": "bbee0e7c3d7abfb6e2539eaf2fb9987b",
    "lsun_church": "eb619b8a5ab95ef80f94ce8a5488dae3",
}

def download(url, local_path, chunk_size = 1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream = True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total = total_size, unit = "B", unit_scale = True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size = chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()

def get_ckpt_path(name, root=None, check=False):
    assert name in URL_MAP
    cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    root = root if root is not None else os.path.join(cachedir, "diffusion_models_converted")
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path)==MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5==MD5_MAP[name], md5
    return path
