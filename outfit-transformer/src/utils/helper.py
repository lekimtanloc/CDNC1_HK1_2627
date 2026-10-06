from __future__ import print_function
import argparse
import torch
import os,sys,random,time
import shutil
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import faiss



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def stack_np(list_of_lists, ax='v'):
    stacked_lists = [np.stack(out, 0) for out in list_of_lists]
    # print('shape stacked list :{}'.format(stacked_lists[0].shape))

    if ax == 'v':
        new_stacked_lists = np.vstack(tuple(stacked_lists))
    elif ax == 'h':
        new_stacked_lists = np.hstack(tuple(stacked_lists))

    # print('final shape of stacked lists :{}'.format(new_stacked_lists.shape))

    return new_stacked_lists


def get_exp_name(args):

    args.dataset = 'polyvore'
    args.name = '{}-{}-{}-{}-bs-{}-lr-p{}-adam{}-{}'.format(args.dataset, args.model_name, args.transform, str(args.resize), str(args.batch_size) , str(args.lr).split('.')[-1], str(args.gamma).split('.')[-1], args.name)
    print('Experiment name : {}'.format(args.name))


def save_script(filename='run_train.sh', experiment='exp'):
    """Saves script to disk"""
    directory = "runs/%s/"%(experiment)
    if not os.path.exists(directory):
        os.makedirs(directory)
    source = 'src/{}'.format(filename)
    if os.path.exists(source):
        shutil.copy2(source, directory + filename)
    else:
        print('warning: reproducibility script is absent; skipping {}'.format(source))

def init_experiment(args, task='cp'):
    # get exp name
    get_exp_name(args)
    # save script
    filename = 'run_train_{}.sh'.format(task)
    save_script(filename, args.name)


def save_checkpoint(state, is_best, experiment, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/"%(experiment)
    if not os.path.isdir(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        print('>>>>>> Saving checkpoint >>>>')
        shutil.copyfile(filename, directory + 'model_best.pth.tar')

def compatibility_auc(targets, probs):
        """ Returns the area under a roc curve for the compatibility
            task
            probs: List of batch tensores of precomputed probabilities of outfits being compatible or non-compatible
            targets: List of batch tensors of ground truth labels for compatibility of outfits
        """
        targets = torch.cat(targets, 0)
        # print('targets shape : {}'.format(targets.shape))
        targets = targets.squeeze().cpu().numpy()

        probs = torch.cat(probs, 0)
        # print('probs shape : {}'.format(probs.shape))
        probs = probs.squeeze().cpu().numpy()


        auc = roc_auc_score(targets, probs)
        return auc


def build_index(
    features: np.ndarray,
    type: str = 'cpu'):
    d = features.shape[1]  # dimension

    if type == 'cpu':
        index= faiss.IndexFlatL2(d)
        # index = faiss.index_factory(d, "Flat", faiss.METRIC_INNER_PRODUCT)
        index.add(features)  # add vectors to the index
        

    elif type == 'gpu':
        res = faiss.StandardGpuResources()  # use a single GPU
        index = faiss.IndexFlatL2(d)  # build the index
        index = faiss.index_cpu_to_gpu(res, 0, index)
        
        index.add(features)  # add vectors to the index
        

    else:
        raise Exception("Unsupported type")
    return index


def compute_recall(neighbors, true_neighbors):
    total = 0
    for gt_row, row in zip(true_neighbors, neighbors):
        total += np.intersect1d(gt_row, row).shape[0]
    return total/true_neighbors.size
