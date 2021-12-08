import trimesh
import os
import numpy as np
import argparse
from joblib import Parallel, delayed

# not used
def job_check_loadable(p):
    try:
        mesh = trimesh.load(p)
    except:
        return os.path.split(p)[1]

def job_convert(p, src_path, dest_path):

    mesh = trimesh.load(p)
    append_str = p.replace(src_path+'/','').replace('.off','.obj')
    dest_p = os.path.join(dest_path, append_str) 
    dr = os.path.split(dest_p)[0] 

    if not os.path.exists(dr):
        os.makedirs(dr)
     
    scale = trimesh.transformations.scale_matrix(1/np.max(mesh.extents), (0,0,0))
    mesh.apply_transform(scale)
   
    mesh.export(dest_p)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, help='Path to downloaded ModelNet (ModelNet40_aligned_fixed_off)')
    parser.add_argument('--dest_path', type=str, help='Path to where to output .off converted to .obj')
    args = parser.parse_args()
    
    if not os.path.exists(args.dest_path):
        os.makedirs(args.dest_path)

    all_paths = []
    categories = os.listdir(args.src_path)
        
    for category in sorted(categories):
        
        train_path = os.path.join(args.src_path, category, 'train')
        test_path = os.path.join(args.src_path, category, 'test')
        
        train_obj_paths = [os.path.join(train_path, x) for x in os.listdir(train_path)]
        test_obj_paths = [os.path.join(test_path, x) for x in os.listdir(test_path)]
            
        paths = train_obj_paths + test_obj_paths
            
        all_paths.extend(paths)
    
    all_paths = [(p, args.src_path, args.dest_path) for p in all_paths]

    bad_objects = Parallel(n_jobs=12, backend='loky', verbose=10)(delayed(job_convert)(*tpl) for tpl in all_paths)
    
    bad_objects = [x for x in bad_objects if x is not None]
    print("Bad objects:", bad_objects)

if __name__ == "__main__":
    main()
