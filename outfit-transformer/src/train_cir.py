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
from models.CIROutfitTransformer import CIROutfitTransformer, CIRMLP


from utils.loss import SetWiseOutfitLoss, SetWiseOutfitLossHard
from data.cir_outfit_dataset import OutfitDataset, MyCollator
from utils.helper import str2bool, stack_np, init_experiment, save_checkpoint
from data.transforms import ScaleLong
from PIL import Image

from sklearn.metrics import roc_auc_score

# Device configuration (set after parsing args)
device = torch.device('cpu')


# Training settings

parser = argparse.ArgumentParser(description='Outfit Transformer in Pytorch')

#training loop
parser.add_argument('--batch_size', type=int, default=50, metavar='N', help='input batch size for training (default: 50)')
parser.add_argument('--epochs', type=int, default=30, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR', help='learning rate (default: 1e-5)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD', help='weight decay (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.5, help='Focal loss parameter')
parser.add_argument('--hard_loss', type=str2bool, default=False, help='add the hard negatives term to the loss')


parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=250, metavar='N', help='how many batches to wait before logging training status')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='CIR_version', type=str, help='suffix name of the experiment')
parser.add_argument('--polyvore_split', default='nondisjoint', type=str, help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--pretrained', default='', type=str)
parser.add_argument('--subset_size', type=int, default=None, help='Optional cap on dataset size for quick runs')

# dataset and transform
parser.add_argument('--datadir', default='data', type=str, help='directory of the polyvore outfits dataset (default: data)')
parser.add_argument('--resize', type=int, default=112, help='Resize transform arg')
parser.add_argument('--transform', default='scale-long', type=str, help='type of image transform')
parser.add_argument('--use_finegrained', type=str2bool, default=False, help='use finegrained categories for negative sampling')
parser.add_argument('--online_neg_sampling', type=str2bool, default=False, help='use online negative sampling, otherwise fixed')
parser.add_argument('--saved_fixed_neg_samples', type=str2bool, default=False, help='use precomputed nagtive sampling')
parser.add_argument('--train_online_sampling', type=str2bool, default=False, help='use fixed neg samples if false for train set (default:true)')


parser.add_argument('--curr_start', type=int, default=15, help='starting epoch of fine training phase')


# model architecture
parser.add_argument('--model_name', type=str, default='resnet18', help='backbone model')
parser.add_argument('--text_feature_dim', type=int, default=6000, help='precomputed text embedding dimension')
parser.add_argument('--dim_embed', type=int, default=64)
parser.add_argument('--mha_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=2048)
parser.add_argument('--dropout_prob', type=float, default=0.1)
parser.add_argument('--n_layers', type=int, default=6)
parser.add_argument('--cls_hidden', type=int, default=2048)
parser.add_argument('--mlp_hidden', type=int, default=2048)
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
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    fn = os.path.join(args.datadir, 'item_metadata.json')
    meta_data = json.load(open(fn, 'r'))

    # init experiment
    init_experiment(args, task='cir')
    
    # get data splits
    start_time = time.time()
    print('bs : {}'.format(args.batch_size))
    train_loader, valid_loader, test_loader, fitb_test_loader = make_data(meta_data)
    print('dalaloader process time in minutes: ',(time.time()-start_time)/60)
    
    # make model
    model = make_model()


    print('$$ Initialize the loss function')

    if args.hard_loss:
        criterion = SetWiseOutfitLossHard().to(device)
    else:
        criterion = SetWiseOutfitLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Optimization loop
    best_acc = 0
    epoch_start_time = time.time()
    for epoch in range(args.epochs):

        if epoch >= args.curr_start:
            print('$$ FINE-GRAINED TRAINING MODE')
            print('-----------------------------------------\n')
            train_loader.dataset.args.use_finegrained=True
            
        print("Epoch {}\n----------------------------------".format(epoch+1))
        # start_time = time.time()
        train(train_loader, model, criterion, optimizer)
        # print('one epoch process time: ',time.time()-start_time)

        # start_time = time.time()
        accuracy = eval(valid_loader, model, criterion)
        # print('one epoch eval process time: ',time.time()-start_time)

        # update learning rate : https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        scheduler.step()

        is_best = accuracy > best_acc
        best_acc = max(accuracy, best_acc)
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_accc': best_acc,
            'optimizer':optimizer.state_dict()
        }, is_best, args.name)

    train_time = (time.time()-epoch_start_time)/60
    print('time until done training: {} minutes'.format((time.time()-epoch_start_time)/60))

    
    # Evaluate the best model using the testset
    start_time = time.time()
    checkpoint = torch.load('runs/%s/'%(args.name) + 'model_best.pth.tar', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])

    # Test best model for random positive/ negative samples for retrieval
    accuracy = test(test_loader, model)

    fitb_accuracy = test(fitb_test_loader, model)
 
    filename =  'reports/{}/'.format(args.name) + 'performance.txt'

    
    path = Path(filename)
    if not os.path.exists(path.parent):
        os.makedirs(path.parent)
    with open(filename, 'w') as f:
        f.write('\n Test set:  Accuracy (random sampling 10 neg): {:.4f} , FITB accuracy: {} , Test time (H): {} , train time (min): {} , best epoch: {}\n'.format(accuracy, fitb_accuracy, (time.time()-start_time)/1440, train_time/1440, checkpoint['epoch']))
    
    print('Test done: {} minutes'.format((time.time()-start_time)/60))


    
def make_data(meta_data):
    normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])
    use_pinned = args.cuda or getattr(args, 'mps', False)
    kwargs = {'num_workers': 2, 'pin_memory': True} if use_pinned else {}

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
    print('$$ VALID SPLIT')
    print('-----------------------------------------\n')
    valid_ds = OutfitDataset(args, 'valid', meta_data, args.text_feature_dim, transform=valid_transform)
    if args.subset_size:
        valid_ds = Subset(valid_ds, range(min(args.subset_size, len(valid_ds))))
    valid_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=args.batch_size, shuffle=False, collate_fn=my_collator, **kwargs)

    print('$$ TRAIN SPLIT')
    print('-----------------------------------------\n')
    train_ds = OutfitDataset(args, 'train', meta_data, args.text_feature_dim, transform=train_transform)
    if args.subset_size:
        train_ds = Subset(train_ds, range(min(args.subset_size, len(train_ds))))
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size, shuffle=True, collate_fn=my_collator, **kwargs)
    
    print('$$ TEST SPLIT')
    print('-----------------------------------------\n')
    test_ds = OutfitDataset(args, 'test', meta_data, args.text_feature_dim, transform=test_transform)
    if args.subset_size:
        test_ds = Subset(test_ds, range(min(args.subset_size, len(test_ds))))
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size, shuffle=False, collate_fn=my_collator, **kwargs)

    print('$$ TEST SPLIT (for FITB)')
    print('-----------------------------------------\n')
    fitb_ds = OutfitDataset(args, 'test', meta_data, args.text_feature_dim, transform=test_transform, phase='test')
    if args.subset_size:
        fitb_ds = Subset(fitb_ds, range(min(args.subset_size, len(fitb_ds))))
    fitb_test_loader = torch.utils.data.DataLoader(
        fitb_ds,
        batch_size=args.batch_size, shuffle=False, collate_fn=my_collator, **kwargs)
    

    return train_loader, valid_loader, test_loader, fitb_test_loader
    # return valid_loader, valid_loader, valid_loader, valid_loader


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
    cp_model = OutfitTransformer(image_text_encoders=image_text_encoders, 
                            transformer_encoder=trans_encoder, 
                            classification=cls_head)
    
    print('$$ Load CP pretrained OutfitTransformer model')
    print('-----------------------------------------')
    checkpoint = torch.load('runs/%s/'%(args.pretrained) + 'model_best.pth.tar', map_location=device, weights_only=False)
    cp_model.load_state_dict(checkpoint['state_dict'])
    
    
    # MLP
    print('$$ Make MLP module')
    print('-----------------------------------------')
    cir_mlp = CIRMLP(d_model=args.dim_embed*2, n_hidden=args.mlp_hidden)
    
    
    print('$$ Make CIROutfitTransformer model')
    print('-----------------------------------------')
    model = CIROutfitTransformer(cp_pretrained_transformer=cp_model, 
                            mlp=cir_mlp)
    
    model = model.to(device)

    return model


def train(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (images, descriptions, has_text_ls, item_types, outfit_lens, pos_items, neg_samples_ids, neg_samples_embeds, pos_sample_embed) in enumerate(dataloader):

        # stack all the outfits in the format of [max_outfit_len * batch_size, 3, 112, 112]
        stacked_images = [torch.stack(out, 0) for out in images]

        new_stacked_images = torch.cat(stacked_images, 0)

        # print('new_stacked_images shape : {}'.format(new_stacked_images.shape))
        # print(new_stacked_images[0,:,:,:])


        # stack all the descriptions in the format of [max_outfit_len * batch_size, dim_embed]
        new_stacked_descs = stack_np(descriptions)

        # print('new_stacked_descs shape : {}'.format(new_stacked_descs.shape))

        # stack all the binary indicators in the format of [max_outfit_len * batch_size, dim_embed]
        new_stacked_hastxt = stack_np(has_text_ls, 'h')

        images = new_stacked_images.to(device)
        descs = torch.from_numpy(new_stacked_descs).to(device)
        has_text = (torch.from_numpy(new_stacked_hastxt)).to(device)

        output = model(images, descs, outfit_lens, len(pos_items))

        # # compute loss and backward pass
        loss = 0.0
        for i in range(output.shape[0]):
            target_embedding= output[i,:]
            pos_embedding = torch.from_numpy(pos_sample_embed[i]).to(device)
            neg_embeddings = torch.from_numpy(neg_samples_embeds[i]).to(device)
            loss += criterion(target_embedding, pos_embedding, neg_embeddings, device)


        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 400 ==0:
            loss, iteration = loss.item(), batch * output.shape[0]
            print('loss: {:.5f} iteration: {}/{}'.format(loss, iteration, size))

            break


def eval(dataloader, model, criterion):
    model.eval()
    size = len(dataloader.dataset)
    print('size of the dataset : {}'.format(size))
    num_batches = len(dataloader)
    test_loss, total, n_questions = 0, 0, 0
    with torch.inference_mode():
        for batch, (images, descriptions, has_text_ls, item_types, outfit_lens, pos_items, neg_samples_ids, neg_samples_embeds, pos_sample_embed) in enumerate(dataloader):
            # stack all the outfits in the format of [max_outfit_len * batch_size, 3, 112, 112]
            stacked_images = [torch.stack(out, 0) for out in images]

            new_stacked_images = torch.cat(stacked_images, 0)


            # stack all the descriptions in the format of [max_outfit_len * batch_size, dim_embed]
            new_stacked_descs = stack_np(descriptions)

            # stack all the binary indicators in the format of [max_outfit_len * batch_size, dim_embed]
            new_stacked_hastxt = stack_np(has_text_ls, 'h')


            images = new_stacked_images.to(device)
            descs = torch.from_numpy(new_stacked_descs).to(device)
            has_text = (torch.from_numpy(new_stacked_hastxt)).to(device)

            output = model(images, descs, outfit_lens, len(pos_items))

            # # compute loss
            loss = 0.0
            for i in range(output.shape[0]):
                target_embedding= output[i,:]
                pos_embedding = torch.from_numpy(pos_sample_embed[i]).to(device)
                neg_embeddings = torch.from_numpy(neg_samples_embeds[i]).to(device)
                loss += criterion(target_embedding, pos_embedding, neg_embeddings, device)

            test_loss += loss.item()
            
            # compute ditances of target embeddings from postivie and negative samples
            
            pdist = nn.PairwiseDistance(p=2)
            
            for i in range(output.shape[0]):
                
                
                pos_embedding = torch.from_numpy(pos_sample_embed[i]).unsqueeze(0)
                # print('shape pos : {}'.format(pos_embedding.shape))
                # print('pos : {}'.format(pos_embedding))
                neg_embeddings = torch.from_numpy(neg_samples_embeds[i])

                nb_neg_samples = neg_embeddings.shape[0]
                is_correct = np.zeros(nb_neg_samples+1, bool)
                is_correct[0] = True

                # concatenate positive embedding and set of negative embeddings
                answers = torch.cat([pos_embedding, neg_embeddings]).to(device)
                # print('shape answ : {}'.format(answers.shape))
                # print('answ : {}'.format(answers))

                target_embedding= output[i,:]
                # print('shape target emb : {}'.format(target_embedding.shape))
                # print('target emb : {}'.format(target_embedding))
                # repeat the target embedding to match the size of negative samples embeddings
                t = target_embedding.repeat(nb_neg_samples+1,1)
                # print('t shape after repeat : {}'.format(t.shape))
                # print('t : {}'.format(t))

                score = pdist(t, answers)
                # print('score shape : {}'.format(score.shape))
                # print('score  : {}'.format(score))

                score = score.cpu().numpy()
                # print('test correctness : {}'.format(is_correct[np.argmin(score)]))
                total += is_correct[np.argmin(score)] 
                # print('total for iter {} bs {}'.format(i, batch))
                # print('total : {}'.format(total))
                n_questions += 1

            if batch % 400==0:
                iteration = batch * output.shape[0]
                print('iteration: {}/{}'.format(iteration, size))

            # break

    test_loss /= num_batches
    print('effective # of the dataset : {}'.format(n_questions))
    total/= n_questions


    print('Accuracy : {:.4f}, Avg loss: {:.8f}'.format(100*total, test_loss))
    return total


def test(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    print('size of the dataset : {}'.format(size))
    num_batches = len(dataloader)
    total, n_questions  = 0, 0
    with torch.inference_mode():
        for batch, (images, descriptions, has_text_ls, item_types, outfit_lens, pos_items, neg_samples_ids, neg_samples_embeds, pos_sample_embed) in enumerate(dataloader):
            # stack all the outfits in the format of [max_outfit_len * batch_size, 3, 112, 112]
            stacked_images = [torch.stack(out, 0) for out in images]

            new_stacked_images = torch.cat(stacked_images, 0)


            # stack all the descriptions in the format of [max_outfit_len * batch_size, dim_embed]
            new_stacked_descs = stack_np(descriptions)

            # stack all the binary indicators in the format of [max_outfit_len * batch_size, dim_embed]
            new_stacked_hastxt = stack_np(has_text_ls, 'h')


            images = new_stacked_images.to(device)
            descs = torch.from_numpy(new_stacked_descs).to(device)
            has_text = (torch.from_numpy(new_stacked_hastxt)).to(device)

            # print('images shape : {}'.format(images.shape))
            # print('descs shape : {}'.format(descs.shape))
            # print(len(pos_items))


            output = model(images, descs, outfit_lens, len(pos_items))
            
            # compute ditances of target embeddings from postivie and negative samples
            pdist = nn.PairwiseDistance(p=2)
            
            for i in range(output.shape[0]):
                
                pos_embedding = torch.from_numpy(pos_sample_embed[i]).unsqueeze(0)
                # print('shape pos : {}'.format(pos_embedding.shape))
                neg_embeddings = torch.from_numpy(neg_samples_embeds[i])

                nb_neg_samples = neg_embeddings.shape[0]
                is_correct = np.zeros(nb_neg_samples+1, bool)
                is_correct[0] = True

                # concatenate positive embedding and set of negative embeddings
                answers = torch.cat([pos_embedding, neg_embeddings]).to(device)
                # print('shape answ : {}'.format(answers.shape))

                target_embedding= output[i,:]
                # repeat the target embedding to match the size of negative samples embeddings
                t = target_embedding.repeat(nb_neg_samples+1,1)
                # print('t shape after repeat : {}'.format(t.shape))

                score = pdist(t, answers)
                # print('score shape : {}'.format(score.shape))
                # print('score  : {}'.format(score))

                score = score.cpu().numpy()
                # print('test correctness : {}'.format(is_correct[np.argmin(score)]))
                total += is_correct[np.argmin(score)] 
                # print('total for iter {} bs {}'.format(i, batch))
                # print('total : {}'.format(total))
                n_questions += 1


            # print('total : {}'.format(total))

            if batch % 400==0:
                iteration = batch * output.shape[0]
                print('iteration: {}/{}'.format(iteration, size))

                # break 

    print('effective # of the dataset : {}'.format(n_questions))
    # print('total over size {}'.format(total/size))
    total/= n_questions

    print('Accuracy : {:.4f}'.format(100*total))
    return total


if __name__ == '__main__':
    main()    
