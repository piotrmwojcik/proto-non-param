import os
import shutil
from PIL import Image
import Augmentor

def is_ok_image(fp: str, min_side: int = 64) -> bool:
    try:
        with Image.open(fp) as im:
            im.verify()  # quick corruption check
        with Image.open(fp) as im:
            w, h = im.size
        return min(w, h) >= min_side
    except Exception:
        return False

IMG_EXTS = (".jpg", ".jpeg", ".png")

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

datasets_root_dir = '/home/pwojcik/CUB_200_2011/'
dir = datasets_root_dir + 'train_cropped/'
target_dir = datasets_root_dir + 'train_cropped_augmented_2/'

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]


for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]

    bad_dir = os.path.join(fd, "_bad")
    os.makedirs(bad_dir, exist_ok=True)
    os.makedirs(tfd, exist_ok=True)

    # --- move bad files out of fd so Augmentor won't see them ---
    for fn in os.listdir(fd):
        fp = os.path.join(fd, fn)
        if not os.path.isfile(fp):
            continue
        if not fn.lower().endswith(IMG_EXTS):
            continue

        if not is_ok_image(fp, min_side=64):
            dst = os.path.join(bad_dir, fn)
            # avoid overwrite collisions
            if os.path.exists(dst):
                base, ext = os.path.splitext(fn)
                dst = os.path.join(bad_dir, f"{base}__dup{ext}")
            shutil.move(fp, dst)

    # --- run augmentation on the cleaned folder ---
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)

    p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
    p.flip_left_right(probability=0.5)
    p.sample(10, multi_threaded=False)

    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=1, magnitude=0.2)
    p.flip_left_right(probability=0.5)
    p.sample(10, multi_threaded=False)

    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    p.sample(10, multi_threaded=False)

    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    p.flip_left_right(probability=0.5)
    p.sample(10, multi_threaded=False)