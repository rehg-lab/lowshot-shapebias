import os
import torch
import json
import numpy as np
import torch
import warnings

import lssb.data.transform as lssb_t

from torchvision import transforms

from PIL import Image

# GLOBALS
N_VIEWS = 25
VALID_MODALITIES = ['image', 'ptcld', 'image+ptcld', 
                    'image+shape_embedding', 'image+shape_embedding+ptcld']

root_dir='./data/TOYS4K'

class Toys4K(torch.utils.data.Dataset):
     
    def __init__(self, 
            split='train', 
            modality=None, 
            use_aug=True, 
            extra_args=None):
   
        self.BASE_PATH = root_dir
        self.extra_args = extra_args
        self.use_aug = use_aug
        self.modality = modality
       
        self.LOWSHOT_JSON_PATH = '{}/toys4k_lowshot_train_dict.json'.format(root_dir)
        
        if split not in ['train', 'val', 'test']:
            raise NameError("<split> has to be one of ['train', 'val', 'test']")
        
        if not modality in VALID_MODALITIES:
            raise NameError("<modality> has to be one of {}".format(VALID_MODALITIES))
       
        with open(self.LOWSHOT_JSON_PATH, 'r') as f:
            self.data_dict = json.load(f)[split]

        self.classes = sorted(self.data_dict.keys())

        print(split, 'dataset has', self.classes)

        # creating list of (category, instance) pairs and labels
        self.objects, self.labels = self.make_dataset()
        
        ### defining dataset transforms for images
        if use_aug:  
            self.img_transforms = transforms.Compose(
                [   
                    transforms.RandomApply([lssb_t.ModelNetImageAug()],p=0.5),
                    transforms.Resize(224), 
                    transforms.ToTensor()
                ]
            )
        else:
            self.img_transforms = transforms.Compose(
                [   
                    transforms.Resize(224), 
                    transforms.ToTensor()
                ]
            )
        
        ### defining dataset transforms for point clouds
        if use_aug and 'ptcld' in modality:
                
            if self.extra_args['use_random_SO3_rotation']:
                rot_trans = lssb_t.PointcloudSO3Rotate()
            else:
                rot_trans = lssb_t.PointcloudRotate(axis=np.array([0, 1, 0]))
        

            self.pc_transforms = transforms.Compose(
                [
                    lssb_t.PointcloudToTensor(),
                    lssb_t.PointcloudScale(),
                    rot_trans,
                    lssb_t.PointcloudRotatePerturbation(),
                    lssb_t.PointcloudTranslate(),
                    lssb_t.PointcloudJitter(),
                    lssb_t.PointcloudRandomInputDropout(),
                ]
            )

        else:
            self.pc_transforms = transforms.Compose(
                [
                    lssb_t.PointcloudToTensor()
                ]
            ) 


    def make_dataset(self):
    
        objects = []
        missing_objects = []
        labels = []
        
        for cls in self.classes:
            class_objects = self.data_dict[cls]
            
            for obj in class_objects:
                
                obj_tpl = (cls, obj)
                obj_ptcld_path = os.path.join(self.BASE_PATH, 
                                              'pointclouds', 
                                              cls, 
                                              obj, 
                                              'pc10K.npz')
                
                obj_img_path = os.path.join(
                    self.BASE_PATH, 
                    'renders',
                    cls,
                    obj)

                exists = os.path.exists(obj_ptcld_path) and os.path.exists(obj_img_path)
                
                if exists:
                    objects.append(obj_tpl)
                    labels.append(self.classes.index(cls))

                else:
                        missing_objects.append(obj)
        
        if len(missing_objects) > 0:
            print("Missing {} objects".format(len(missing_objects)))
        
        #if training with DGCNN shape (point cloud) embeddings, then load them 
        if 'shape_embedding' in self.modality:
            self.embedding_dict = np.load(
                    os.path.join(self.BASE_PATH, 'features', self.extra_args['feat_dict_file']), 
                    allow_pickle=True
                )['feat_dict'].item()

        return objects, labels
    
    def get_paths(self, cls, obj, view):
        """
            given the object class <cls>, 
            object instance name <obj>,
            and the <view> - integer value from 0 to 25
            return the path for the point cloud and the image
        """

        ptcld_path = os.path.join(
            self.BASE_PATH,
            'pointclouds',
            cls,
            obj,
            'pc10K.npz'
        )
        
        img_path = os.path.join(
            self.BASE_PATH, 
            'renders',
            cls,
            obj,
            'image_output', 
            '{:04d}.png'.format(view)
        )

        
        return {'ptcld':ptcld_path, 
                'image':img_path}

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        view = int(torch.randint(N_VIEWS, (1,)))

        cls, obj = self.objects[idx]
        data = {}

        paths = self.get_paths(cls, obj, view)

        if 'ptcld' in self.modality:
            arr = np.load(paths['ptcld'],allow_pickle=True)['dct'].item()
            point_set = arr['pc']
            
            #hard coding n_points here
            pt_idxs = np.arange(0, len(point_set))
            pt_idxs = np.random.choice(pt_idxs, self.extra_args['num_points'])

            point_set = point_set[pt_idxs, :]
            point_set = lssb_t.pc_normalize(point_set)
            point_set = self.pc_transforms(point_set) 
        
            point_set = point_set.T

            # model input is [3, num_points] where 3 is [X,Y,Z]
            data['ptcld'] = point_set
        
        if 'image' in self.modality:

            with open(paths['image'], 'rb') as f: 
                img = Image.open(f).convert('RGB')
            if self.img_transforms is not None:
                img = self.img_transforms(img)
            data['image'] = img
    
        if 'embedding' in self.modality:
            # load point cloud feature embedding
            arr = self.embedding_dict[(cls, obj)]
            
            feat_idx = np.random.choice(len(arr))
            arr = arr[feat_idx]

            data['gt_embed'] = torch.FloatTensor(arr)

        data['labels'] = self.labels[idx]
        data['metadata'] = (cls, obj)

        return data

if __name__ == "__main__":
    extra_args = {
        'use_random_SO3_rotation':True,
        'num_points':1024
        }
    
    dataset = Toys4K(split='train', modality='ptcld', extra_args=extra_args)
    dataset = Toys4K(split='val', modality='ptcld', extra_args=extra_args)
    dataset = Toys4K(split='test', modality='ptcld', extra_args=extra_args)
    
    dataset = Toys4K(split='train', modality='image')
    dataset = Toys4K(split='val', modality='image')
    dataset = Toys4K(split='test',modality='image')

    data_item = dataset.__getitem__(0)
