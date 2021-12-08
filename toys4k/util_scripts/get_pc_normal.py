import os
import json
import trimesh
import numpy as np
import argparse

from joblib import Parallel, delayed

def job(arg):
    obj_path, out_path = arg

    mesh = trimesh.load(obj_path)
    pc, idx = trimesh.sample.sample_surface(mesh, 10000)
    normals = mesh.face_normals[idx]

    out_dict_1 = {
            'pc':pc,
            'normals':normals
            }
    
    np.savez_compressed(os.path.join(out_path, 'pc10K.npz'), dct=out_dict_1)

def main():
     
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--toys_obj_path', type=str, required=True)
    parser.add_argument('--toys_json_path', type=str, required=True)

    dest_dir = './pointclouds'

    args = parser.parse_args()

    toys_path = args.toys_obj_path
    json_path = args.toys_json_path
   
    path_list = []
    with open(json_path,'r') as f:
        data_dict = json.load(f)

    for categ, objs in data_dict.items():
        
        for obj in objs:
            
            obj = obj.split('/')[0]
            obj_path = os.path.join(toys_path, categ, obj, 'mesh.obj')
            
            if not os.path.exists(obj_path):
                print('not found', obj_path)
                continue
            
            output_path = os.path.join(dest_dir, categ, obj)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            path_list.append((obj_path, output_path))
        
    Parallel(n_jobs=20, verbose=10, backend='loky')(delayed(job)(arg) for arg in path_list)

if __name__ == "__main__":
    main()
