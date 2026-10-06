from __future__ import print_function
import argparse
import os,sys,random,time
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
from torch.utils.data import Subset

from models.models import BackModel
from models.ImageTextEncoders import ImageTextEncoder
from models.OutfitTransformer import OutfitTransformer, ClassificationHead

from utils.loss import FocalLoss, SigmoidFocalLoss
from data.outfit_dataset import OutfitDataset, MyCollator
from utils.helper import str2bool, stack_np, init_experiment, save_checkpoint, compatibility_auc
from data.transforms import ScaleLong
from PIL import Image


from sklearn.metrics import roc_auc_score

# Device configuration (filled in after parsing args)
device = torch.device('cpu')


# Training settings

parser = argparse.ArgumentParser(description='Outfit Transformer in Pytorch')

#training loop
parser.add_argument('--batch_size', type=int, default=50, metavar='N', help='input batch size for training (default: 50)')
parser.add_argument('--epochs', type=int, default=15, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 1e-5)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD', help='weight decay (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.5, help='Focal loss parameter')


parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=250, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='CP_version', type=str, help='suffix name of the experiment')
parser.add_argument('--polyvore_split', default='nondisjoint', type=str, help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--subset_size', type=int, default=None, help='Optional cap on dataset size for quick runs')

# dataset and transform
parser.add_argument('--datadir', default='data', type=str, help='directory of the polyvore outfits dataset (default: data)')
parser.add_argument('--resize', type=int, default=112, help='Resize transform arg')
parser.add_argument('--transform', default='scale-long', type=str, help='type of image transform')


# model architecture
parser.add_argument('--model_name', type=str, default='resnet18', help='backbone model')
parser.add_argument('--text_feature_dim', type=int, default=768, help='precomputed text embedding dimension')
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

    # enable cuDNN autotuning only for CUDA
    torch.backends.cudnn.benchmark = args.cuda

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print('gamma : {}'.format(args.gamma))
    print('datadir : {}'.format(args.datadir))
    fn = os.path.join(args.datadir, 'item_metadata.json')
    meta_data = json.load(open(fn, 'r'))

    # init experiment
    init_experiment(args, task='cp')

    # get data splits
    start_time = time.time()
    train_loader, valid_loader, test_loader = make_data(meta_data)
    print('dalaloader process time in minutes: ',(time.time()-start_time)/60)
    
    # make model
    model = make_model()


    print('$$ Initialize the loss function')
    if args.n_classes == 1:
        criterion = SigmoidFocalLoss(gamma=args.gamma).to(device)
    else:
        criterion = FocalLoss(weight=None, gamma=args.gamma).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Optimization loop
    best_acc = 0
    best_auc = 0
    epoch_start_time = time.time()
    for epoch in range(args.epochs):
        print("Epoch {}\n----------------------------------".format(epoch+1))
        # start_time = time.time()
        train(train_loader, model, criterion, optimizer)
        # print('one epoch process time: ',time.time()-start_time)

        # start_time = time.time()
        accuracy, auc = eval(valid_loader, model, criterion)
        # print('one epoch eval process time: ',time.time()-start_time)

        # update learning rate : https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step()

        is_best = auc > best_auc
        best_auc = max(auc, best_auc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_auc': best_auc,
            'optimizer':optimizer.state_dict()
        }, is_best, args.name)
        # is_best = accuracy > best_acc
        # best_acc = max(accuracy, best_acc)
        # save_checkpoint({
        #     'epoch': epoch,
        #     'state_dict': model.state_dict(),
        #     'best_acc': best_acc,
        #     'optimizer':optimizer.state_dict()
        # }, is_best)
    train_time = (time.time()-epoch_start_time)/60
    print('time until done training: {} minutes'.format((time.time()-epoch_start_time)/60))

    
    # Evaluate the best model using the testset
    start_time = time.time()
    checkpoint = torch.load('runs/%s/'%(args.name) + 'model_best.pth.tar', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    accuracy, auc = test(test_loader, model)
 
    filename =  'reports/{}/'.format(args.name) + 'performance.txt'

    
    path = Path(filename)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)
    with open(filename, 'w') as f:
        f.write('\n Test set: Compat AUC: {:.4f} Accuracy: {:.4f}. Test time (H): {} , train time (H): {} , best epoch: {}\n'.format(auc, accuracy, (time.time()-start_time)/1440,train_time/1440, checkpoint['epoch']))
    
    print('Test done: {} minutes'.format((time.time()-start_time)/60))


    
def make_data(meta_data):
    normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    if args.transform == 'scale-and-crop':

        test_transform = transforms.Compose([transforms.Resize(args.resize), transforms.CenterCrop(args.resize), normalize])
        train_transform = transforms.Compose([
                                transforms.Resize(args.resize),
                                transforms.CenterCrop(args.resize),
                                transforms.RandomHorizontalFlip(),
                                normalize
                            ])
        valid_transform = transforms.Compose([transforms.Resize(args.resize), transforms.CenterCrop(args.resize), normalize])
    elif args.transform == 'scale-long':

        test_transform = transforms.Compose([ScaleLong(args.resize, interpolation=Image.Resampling.LANCZOS), normalize])

        train_transform = transforms.Compose([ScaleLong(args.resize, interpolation=Image.Resampling.LANCZOS), transforms.RandomHorizontalFlip(), normalize])

        valid_transform = transforms.Compose([ScaleLong(args.resize, interpolation=Image.Resampling.LANCZOS), normalize])


    my_collator = MyCollator(resize=args.resize, transform=normalize)
    print('$$ TRAIN SPLIT')
    print('-----------------------------------------')
    train_ds = OutfitDataset(args, 'train', meta_data, args.text_feature_dim, transform=train_transform)
    if args.subset_size:
        # draw a random subset so we don't accidentally keep only one class from the head of the dataset
        subset_len = min(args.subset_size, len(train_ds))
        subset_idx = torch.randperm(len(train_ds))[:subset_len]
        train_ds = Subset(train_ds, subset_idx.tolist())
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size, shuffle=True, collate_fn=my_collator, **kwargs)
    
    print('$$ TEST SPLIT')
    print('-----------------------------------------')
    test_ds = OutfitDataset(args, 'test', meta_data, args.text_feature_dim, transform=test_transform)
    if args.subset_size:
        subset_len = min(args.subset_size, len(test_ds))
        subset_idx = torch.randperm(len(test_ds))[:subset_len]
        test_ds = Subset(test_ds, subset_idx.tolist())
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size, shuffle=False, collate_fn=my_collator, **kwargs)
    
    print('$$ VALID SPLIT')
    print('-----------------------------------------')
    valid_ds = OutfitDataset(args, 'valid', meta_data, args.text_feature_dim, transform=valid_transform)
    if args.subset_size:
        subset_len = min(args.subset_size, len(valid_ds))
        subset_idx = torch.randperm(len(valid_ds))[:subset_len]
        valid_ds = Subset(valid_ds, subset_idx.tolist())
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=args.batch_size, shuffle=False, collate_fn=my_collator, **kwargs)

    return train_loader, valid_loader, test_loader


def make_model():
    # Image and text encoder
    image_text_encoders = ImageTextEncoder(model_name=args.model_name, embedding_size=args.dim_embed, text_dim=args.text_feature_dim)

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
    trans_layer = TransformerLayer(d_model=args.dim_embed*2, self_attn=mha, feed_forward=ffn, dropout_prob=args.dropout_prob)
    trans_encoder = TransformerEncoder(layer=trans_layer, n_layers=args.n_layers)

    # Classification Head
    print('$$ Make Classification Head module')
    print('-----------------------------------------')
    cls_head = ClassificationHead(d_model=args.dim_embed*2, n_hidden=args.cls_hidden, n_classes=args.n_classes) 


    print('$$ Make OutfitTransformer model')
    print('-----------------------------------------')
    model = OutfitTransformer(image_text_encoders=image_text_encoders, 
                            transformer_encoder=trans_encoder, 
                            classification=cls_head)
    
    model = model.to(device)

    return model


def prepare_batch(batch, device):
    """Stack variable-length outfits and move tensors to the target device."""
    images, descriptions, has_text_ls, item_types, outfit_lens, gts = batch

    stacked_images = [torch.stack(out, 0) for out in images]
    images_tensor = torch.cat(stacked_images, 0).to(device)

    descs = torch.from_numpy(stack_np(descriptions)).to(device)
    has_text = torch.from_numpy(stack_np(has_text_ls, 'h')).to(device)
    targets = torch.tensor(gts, dtype=torch.long, device=device)

    return images_tensor, descs, has_text, outfit_lens, targets


def train(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, descriptions, has_text_ls, item_types, outfit_lens, gts) in enumerate(dataloader):

        images, descs, has_text, outfit_lens, target = prepare_batch(
            (images, descriptions, has_text_ls, item_types, outfit_lens, gts), device
        )

        output = model(images, descs, outfit_lens, target.shape[0])


        # compute loss and backward pass
        loss = criterion(output, target.float())


        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 ==0:
            loss, iteration = loss.item(), batch * output.shape[0]
            print('loss: {:.5f} iteration: {}/{}'.format(loss, iteration, size))

            break


def eval(dataloader, model, criterion):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, total = 0, 0
    all_gts, all_probs = [], []
    with torch.inference_mode():
        for batch, (images, descriptions, has_text_ls, item_types, outfit_lens, gts) in enumerate(dataloader):
            images, descs, has_text, outfit_lens, target = prepare_batch(
                (images, descriptions, has_text_ls, item_types, outfit_lens, gts), device
            )


            output = model(images, descs, outfit_lens, target.shape[0])
            test_loss += criterion(output, target.float()).item()
            # correct = (output.argmax(1) == target).type(torch.float).sum().item()
            if args.n_classes ==1:
                probs = torch.sigmoid(output)
                pred = torch.round(probs)
            else:
                probs = (F.softmax(output, dim=1)).argmax(1)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability


            correct = pred.eq(target.view_as(pred)).sum().item()
            # print('for # of instances : {} correct predictions : {}'.format(target.shape[0], correct))
            total += correct 

            # collect targets and probabilities 
            all_gts.append(target)
            all_probs.append(probs)

            # break

    test_loss /= num_batches
    total/= size

    # calculate auc
    auc = compatibility_auc(all_gts, all_probs)
    print('AUC : {}, Accuracy : {}, Avg loss: {:.8f}'.format(round(auc, 2), 100*total, test_loss))
    return total, auc


def test(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    total = 0
    all_gts, all_probs = [], []
    with torch.inference_mode():
        for batch, (images, descriptions, has_text_ls, item_types, outfit_lens, gts) in enumerate(dataloader):
            images, descs, has_text, outfit_lens, target = prepare_batch(
                (images, descriptions, has_text_ls, item_types, outfit_lens, gts), device
            )

            output = model(images, descs, outfit_lens, target.shape[0])
            # correct = (output.argmax(1) == target).type(torch.float).sum().item()
            if args.n_classes ==1:
                probs = torch.sigmoid(output)
                pred = torch.round(probs)
            else:
                probs = (F.softmax(output, dim=1)).argmax(1)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability

            correct = pred.eq(target.view_as(pred)).sum().item()
            print('for # of instances : {} correct predictions : {}'.format(target.shape[0], correct))
            total += correct 

            # collect targets and probabilities 
            all_gts.append(target)
            all_probs.append(probs)

            # break

    total/= size

    # calculate auc
    auc = compatibility_auc(all_gts, all_probs)
    print('AUC : {}, Accuracy : {}'.format(round(auc, 2), 100*total))
    return total, auc






if __name__ == '__main__':
    main()    
