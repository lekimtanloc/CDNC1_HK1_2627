from __future__ import print_function
import argparse
import os
import sys
import random
import time
import shutil
import json

import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from models.models import BackModel
from models.ImageTextEncoders import ImageTextEncoder
from models.OutfitTransformer import OutfitTransformer, ClassificationHead

from data.flat_dataset import FlatDataset

from sklearn.metrics import roc_auc_score

# Device configuration (set after parsing args)
device = torch.device('cpu')


# Inference settings
parser = argparse.ArgumentParser(description='Outfit Transformer in Pytorch')

parser.add_argument('--batch_size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    metavar='S', help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true',
                    default=False, help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=250, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--polyvore_split', default='nondisjoint', type=str,
                    help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--pretrained', default='', type=str)

# dataset and transform
parser.add_argument('--datadir', default='data', type=str,
                    help='directory of the polyvore outfits dataset (default: data)')
parser.add_argument('--resize', type=int, default=112,
                    help='Resize transform arg')

# model architecture
parser.add_argument('--model_name', type=str,
                    default='resnet18', help='backbone model')
parser.add_argument('--text_feature_dim', type=int, default=6000,
                    help='precomputed text embedding dimension')
parser.add_argument('--dim_embed', type=int, default=64)
parser.add_argument('--mha_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--dropout_prob', type=float, default=0.1)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--cls_hidden', type=int, default=2048)
parser.add_argument('--n_classes', type=int, default=1)


def select_device(args):
    """Choose the best available device with MPS support for Apple Silicon."""
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    if not args.no_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def main():
    global args, device
    args = parser.parse_args()
    device = select_device(args)
    print('device is {}'.format(device))

    args.cuda = device.type == 'cuda'
    args.mps = device.type == 'mps'

    torch.backends.cudnn.benchmark = args.cuda

    torch.manual_seed(args.seed)
    np.random.seed(1234)             # make reproducible
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        print("$$$ CPU USAGE $$$")

    fn = os.path.join(args.datadir, 'item_metadata.json')
    meta_data = json.load(open(fn, 'r'))

    # get data splits
    start_time = time.time()
    train_loader, valid_loader, test_loader = make_data(meta_data)
    print('dalaloader process time in minutes: ', (time.time()-start_time)/60)

    # make model
    model = make_model()

    # Inference and save splits embeddings to disk
    start_time = time.time()
    checkpoint = torch.load('runs/%s/' %
                                (args.pretrained) + 'model_best.pth.tar', map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['state_dict'])

    inference(valid_loader, model, 'valid')
    inference(test_loader, model, 'test')
    inference(train_loader, model, 'train')

    print('Inference done: {} minutes'.format((time.time()-start_time)/60))


def make_data(meta_data):
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    use_pinned = args.cuda or getattr(args, 'mps', False)
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_pinned else {}

    test_transform = transforms.Compose([transforms.Resize(
        args.resize), transforms.CenterCrop(args.resize), normalize])

    print('$$ VALID SPLIT')
    print('-----------------------------------------')
    valid_loader = torch.utils.data.DataLoader(
        FlatDataset(args, 'valid', meta_data,
                    args.text_feature_dim, transform=test_transform),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    print('$$ TEST SPLIT')
    print('-----------------------------------------')
    test_loader = torch.utils.data.DataLoader(
        FlatDataset(args, 'test', meta_data, args.text_feature_dim,
                    transform=test_transform),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    print('$$ TRAIN SPLIT')
    print('-----------------------------------------')
    train_loader = torch.utils.data.DataLoader(
        FlatDataset(args, 'train', meta_data,
                    args.text_feature_dim, transform=test_transform),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader


def make_model():
    # Image and text encoder
    image_text_encoders = ImageTextEncoder(
        model_name=args.model_name, embedding_size=args.dim_embed, text_dim=args.text_feature_dim)

    # Transformer encoder
    # MHA
    from labml_nn.transformers.mha import MultiHeadAttention
    print('$$ Make MHA module')
    print('-----------------------------------------')
    mha = MultiHeadAttention(heads=args.mha_heads, d_model=args.dim_embed*2)

    # FFN : position-wise FeedForward Network
    from labml_nn.transformers.feed_forward import FeedForward
    print('$$ Make FFN module')
    print('-----------------------------------------')
    ffn = FeedForward(d_model=args.dim_embed*2, d_ff=args.d_ff)

    # Tranformer encoder
    from models.TransformerEncoder import TransformerEncoder, TransformerLayer
    print('$$ Make Transformer Encoder module')
    print('-----------------------------------------')
    trans_layer = TransformerLayer(
        d_model=args.dim_embed*2, self_attn=mha, feed_forward=ffn, dropout_prob=args.dropout_prob)
    trans_encoder = TransformerEncoder(
        layer=trans_layer, n_layers=args.n_layers)

    # Classification Head
    print('$$ Make Classification Head module')
    print('-----------------------------------------')
    cls_head = ClassificationHead(
        d_model=args.dim_embed*2, n_hidden=args.cls_hidden, n_classes=args.n_classes)

    print('$$ Make OutfitTransformer model')
    print('-----------------------------------------')
    model = OutfitTransformer(image_text_encoders=image_text_encoders,
                              transformer_encoder=trans_encoder,
                              classification=cls_head)

    model = model.to(device)

    return model


def inference(dataloader, model, sp):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    embeddings = []
    has_text_ls = []
    im_ids_ls = []
    split_embed = {}
    with torch.inference_mode():
        for batch, (im_ids, images, descs, has_text) in enumerate(dataloader):
            # print(type(im_ids))
            # print(type(has_text))
            im_ids_ls.append(im_ids)
            has_text_ls.append(has_text)

            images = images.to(device)
            descs = descs.to(device)
            # print('type of has descs : {}'.format(type(has_text)))
            # print('has text : {}'.format(has_text))

            output = model.image_text_encoders(images, descs)
            embeddings.append(output.cpu())

            if batch % 100 == 0:
                iteration = batch * output.shape[0]
                print('iteration: {}/{}'.format(iteration, size))

                # print('sanity : im id = {}'.format(im_ids[0]))
                # print('sanity : embed = {}'.format(output[0,:10]))

            # print('shape of embeddings : {}'.format(output.shape))

    embeddings = torch.cat(embeddings, 0)
    has_text_ls = torch.cat(has_text_ls, 0)
    im_ids_ls = [im for ll in im_ids_ls for im in ll]

    embeddings = embeddings.numpy()
    has_text_ls = has_text_ls.numpy()
    im_ids_ls = np.array(im_ids_ls)

    split_embed['im_ids'] = im_ids_ls
    split_embed['has_text'] = has_text_ls
    split_embed['embed'] = embeddings

    print(split_embed['im_ids'].shape)
    print(split_embed['has_text'].shape)
    print(split_embed['embed'].shape)

    filename = 'embeddings/{}/'.format(args.pretrained) + \
        '{}_embeddings.npy'.format(sp)

    path = Path(filename)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)

    np.save(filename, split_embed)


if __name__ == '__main__':
    main()
