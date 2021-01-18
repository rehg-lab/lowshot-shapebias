import torch
import numpy as np


def extract_pc_features(model, loader, cnt, n):
    output_dict = {}

    for idx, batch in enumerate(loader):
        with torch.no_grad():
            print("[{:04d}/{:04d} | {:04d}/{:04d}]".format(
                idx, len(loader), cnt, n), end='\r')
            ptcld = batch['ptcld']
            ptcld = ptcld.cuda()
            output = model.net(ptcld)
            
            for feature, cls, obj in zip(output['embed'], *batch['metadata']):
                output_dict[(cls, obj)]=feature.detach().cpu().numpy()
        
    print()

    return output_dict

def get_multiple_features(model, loader, n):
    dicts = [extract_pc_features(model, loader, cnt, n) for cnt in np.arange(n)]

    dict_ = {}
    keys = dicts[0].keys()

    for key in keys:
        dict_[key] = []
        for dct in dicts:
            feats = dct[key]
            dict_[key].append(feats)
        
        dict_[key] = np.array(dict_[key])
    
    return dict_


