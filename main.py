import numpy as np
from tqdm import tqdm

import torch
from transformers import DistilBertTokenizer

import config as CFG
from dataset import CLIPDataset, get_transforms
from CLIP import CLIPModel
from utils import AvgMeter, get_lr
import json

def build_loaders(tokenizer, mode):
    transforms = get_transforms(mode=mode)
    with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
        data = json.load(fp)['images']
    if mode == 'train':
        samples = [x for x in data if x['split'] in ['train', 'restval']]
    elif mode == 'valid':
        samples = [x for x in data if x['split'] == 'val']
    else:
        assert False
    image_paths = []
    captions = []
    for sample in samples:
        for caption_data in sample['sentences']:
            image_paths.append(f'{sample["filepath"]}/{sample["filename"]}')
            captions.append(caption_data['raw'])
    dataset = CLIPDataset(
        image_paths,
        captions,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(CFG.device) for k, v in batch.items() if k != "caption"}
        loss = model(batch)

        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def main():
    print('Loading tokenizer', flush=True)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    print('Loading datasets', flush=True)
    train_loader = build_loaders(tokenizer, mode="train")
    valid_loader = build_loaders(tokenizer, mode="valid")

    print('Loading model', flush=True)
    model = CLIPModel().to(CFG.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
    )
    step = "epoch"

    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "best.pt")
            print("Saved Best Model!")


if __name__ == "__main__":
    main()
