import torch
import numpy as np
import torch.nn.functional as F

from numpy import linalg as LA
from scipy.stats import mode

from lssb.data.sampler import CategoriesSampler
from joblib import Parallel, delayed

"""
functions here are largely borrowed from the original SimpleShot repository
https://github.com/mileyan/simple_shot/
"""

def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]

def metric_prediction_val(gallery, query, train_label, test_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1)
    query = query.view(query.shape[0], -1)
    distance = get_metric(metric_type)(gallery, query)
    predict = torch.argmin(distance, dim=1)
    predict = torch.take(train_label, predict)

    acc = (predict == test_label).float().mean().detach().cpu()

    return acc

def metric_prediction_test(gallery,
            query, 
            train_label, 
            test_label, 
            shot, 
            way,
            train_mean=None, 
            norm_type='CL2N', num_NN=1):

    if norm_type == 'CL2N':
        gallery = gallery - train_mean
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query - train_mean
        query = query / LA.norm(query, 2, 1)[:, None]
    elif norm_type == 'L2N':
        gallery = gallery / LA.norm(gallery, 2, 1)[:, None]
        query = query / LA.norm(query, 2, 1)[:, None]

    gallery = gallery.reshape(way, shot, gallery.shape[-1]).mean(1)
    train_label = train_label[::shot]
    subtract = gallery[:, None, :] - query
    distance = LA.norm(subtract, 2, axis=-1)
    idx = np.argpartition(distance, num_NN, axis=0)[:num_NN]
    nearest_samples = np.take(train_label, idx)
    out = mode(nearest_samples, axis=0)[0]
    out = out.astype(int)
    test_label = np.array(test_label)
    acc = (out == test_label).mean()
    
    return acc

def test_func(feature, label, n_shot, n_way, train_mean, norm_type):
    idx = n_way * n_shot
    
    train_feat = feature[:idx]
    train_label = label[:idx]
    test_feat = feature[idx:]
    test_label = label[idx:]
    
    acc = metric_prediction_test(train_feat, 
                                 test_feat, 
                                 train_label, 
                                 test_label, 
                                 n_shot,
                                 n_way,
                                 train_mean=train_mean,
                                 norm_type=norm_type)

    return acc

def run_lowshot_testing(
        mean_feature, 
        test_feats, 
        test_labels,
        n_iter, 
        n_shot, 
        n_way, 
        n_query):
    
    """
    Compute low-shot testing accuracy without normalization, with normalization
    and with normalization and centering.

    params:
    mean_feature - mean feature vector from training set
    test_feats - [n_episodes, n_shot*n_way + n_query, feature_dim] np array of embeddings
    test_labels - [n_episodes, n_shot*n_way + n_query] np array of labels
    n_iter - number of low-shot iterations
    n_shot - number of low-shot support shots
    n_way - number of low-shot ways
    n_query - number of low-shot queries

    returns:
    out_str: a string with results
    out_dict: a dictionary with results for UN, L2N and CL2N settings of simpleshot
    """

    out_str = "LOW-SHOT TESTING {} iters {} shots {} ways {} queries\n\n".format(
            n_iter,
            n_shot,
            n_way,
            n_query)

    print()
    print()
     
    un_list = Parallel(n_jobs=10, backend='loky', verbose=1)(
                delayed(test_func)
                    (feature=feature,
                     label=label, 
                     n_shot=shot, 
                     n_way=way, 
                     train_mean=mean_feature, 
                     norm_type=norm_type) 
                 for feature,  label in zip(test_feats, test_labels)\
                 for shot in [n_shot] \
                 for way in [n_way] \
                 for norm_type in ['UN'])
    
    l2n_list = Parallel(n_jobs=10, backend='loky', verbose=1)(
                delayed(test_func)
                    (feature=feature, 
                     label=label, 
                     n_shot=shot, 
                     n_way=way, 
                     train_mean=mean_feature, 
                     norm_type=norm_type) 
                 for feature, label in zip(test_feats, test_labels) \
                 for shot in [n_shot] \
                 for way in [n_way] \
                 for norm_type in ['L2N'])
    
    cl2n_list = Parallel(n_jobs=10, backend='loky', verbose=1)(
                delayed(test_func)
                    (feature=feature, 
                     label=label, 
                     n_shot=shot, 
                     n_way=way, 
                     train_mean=mean_feature, 
                     norm_type=norm_type) 
                 for feature, label in zip(test_feats, test_labels) \
                 for shot in [n_shot] \
                 for way in [n_way] \
                 for norm_type in ['CL2N'])

    un_mean, un_conf = compute_confidence_interval(un_list)
    l2n_mean, l2n_conf = compute_confidence_interval(l2n_list)
    cl2n_mean, cl2n_conf = compute_confidence_interval(cl2n_list)
    
    out_str += 'Test:\n'
    out_str += 'feature  UN            L2N           CL2N\n'
    out_str += '{} Shot   {:.4f}({:.4f})  {:.4f}({:.4f})  {:.4f}({:.4f})\n'.format(
        n_shot, 
        un_mean, 
        un_conf,
        l2n_mean, 
        l2n_conf,
        cl2n_mean, 
        cl2n_conf
    )
    
    out_dict = {
            'UN':(un_mean, un_conf),
            'L2N':(l2n_mean, l2n_conf),
            'CL2N':(cl2n_mean, cl2n_conf)
            }
    print(out_str)
    print()
    print()

    return out_str, out_dict

def run_lowshot_validation(embeds, labels, n_shot, n_way, val_metric):
    idx = n_way * n_shot
    
    accs = []
    for embed, label in zip(embeds, labels):
        train_out = embed[:idx]
        train_label = label[:idx]
        test_out = embed[idx:]
        test_label = label[idx:]
        train_out = train_out.reshape(n_way, n_shot, -1).mean(1)
        train_label = train_label[::n_shot]
        
        acc = metric_prediction_val(train_out, test_out, train_label, test_label, val_metric)
        accs.append(acc)

    val_ls_acc = torch.Tensor(accs).mean() 
    
    return val_ls_acc

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

