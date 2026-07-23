"""ReferDataset: loads pre-built .npz batches produced by build_batches.py (ported from kdwonn/SaG data.py)."""
import os
import os.path as osp
import random
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer


def process_caption_bert(caption, tokenizer, drop_prob, train):
    output_tokens = []
    deleted_idx = []
    tokens = tokenizer.basic_tokenizer.tokenize(caption)
    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()
        if prob < drop_prob and train:
            prob /= drop_prob
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                output_tokens.append(sub_token)
    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]
    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target


class ReferDataset(data.Dataset):
    """Dataset that reads pre-built .npz batch files from build_batches.py."""

    def __init__(self, root, splitset, transform=None, drop_prob=0):
        self.root = root
        self.train = splitset == 'train'
        self.transform = transform
        self.set = splitset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.drop_prob = drop_prob
        self.data_list = sorted(glob(osp.join(root, self.set + '_batch', '*')))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        sentence, img_id, image, label, im_name = self.get_raw_item(index)
        if self.transform is not None:
            image = self.transform(image)
        target = process_caption_bert(sentence, self.tokenizer, self.drop_prob, self.train)
        return image, target, index, img_id

    def get_raw_item(self, index):
        datafiles = np.load(self.data_list[index])
        image = Image.fromarray(datafiles["im_batch"]).convert('RGB')
        label = datafiles["mask_batch"]
        sentence = datafiles["sent_batch"][0]
        im_name = str(datafiles['im_name_batch'])
        img_id = osp.basename(self.data_list[index]).split(".")[0].split("_")[-1]
        return sentence, img_id, image, label, im_name


def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, ids, img_ids = zip(*data)
    images = torch.stack(images, 0)
    cap_lengths = torch.tensor([len(cap) for cap in sentences])
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, cap_lengths, ids


def get_image_transform(split_name, img_backbone, crop_size, use_aug):
    if 'vit' in img_backbone:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif 'res' in img_backbone:
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        raise NotImplementedError
    t_list = []
    if split_name == 'train':
        if use_aug:
            t_list = [transforms.RandomResizedCrop(
                size=(crop_size, crop_size), scale=(0.8, 1.0), ratio=(0.8, 1.2))]
        else:
            t_list = [transforms.Resize(size=(crop_size, crop_size))]
    else:
        t_list = [transforms.Resize(size=(crop_size, crop_size))]
    t_end = [transforms.ToTensor(), normalizer]
    return transforms.Compose(t_list + t_end)


def get_loader(data_root, dataset, split, img_backbone, crop_size, use_aug,
               batch_size=32, shuffle=True, num_workers=4):
    """Factory that returns a DataLoader for a given dataset/split."""
    root = osp.join(data_root, dataset)
    batch_dir = osp.join(root, split + '_batch')
    transform = get_image_transform(split, img_backbone, crop_size, use_aug)
    ds = ReferDataset(root=root, splitset=split, transform=transform)
    if len(ds) == 0:
        raise RuntimeError(
            f"No batch files found in: {batch_dir}\n"
            f"  data_root={data_root!r}  dataset={dataset!r}  split={split!r}\n"
            f"  Run build_batches.py to generate .npz files."
        )
    return data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
    )


def get_train_loader(args):
    return get_loader(
        data_root=args.data_root,
        dataset=args.dataset,
        split=args.data_split,
        img_backbone=args.img_backbone,
        crop_size=args.crop_size,
        use_aug=args.use_aug,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )


def get_test_loader(args, split='val'):
    return get_loader(
        data_root=args.data_root,
        dataset=args.dataset,
        split=split,
        img_backbone=args.img_backbone,
        crop_size=args.crop_size,
        use_aug=False,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1,
    )
