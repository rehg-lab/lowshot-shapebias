from pytorch_lightning import Trainer
from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
from pytorch_lightning.core.saving import update_hparams
import os
import glob
import yaml
import lssb.lowshot.models as lowshot_models

def flatten_dict(dct):
    """
        If a dictionary has sub-dictionaries, flatten 
        it into one dictionary
    """
    out_dct = {}

    for key, value in dct.items():
        assert key not in out_dct.keys()

        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                assert sub_key not in out_dct.keys()

                out_dct[sub_key] = sub_value
        else:
            out_dct[key] = value
    
    return out_dct

def main():
    
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument('--log_dir', type=str,
                        help="Directory where the experiment is located - log_dir in the .yaml")
    parser.add_argument('--name', type=str,
                        help="Experiment name - exp_name in the .yaml")
    parser.add_argument('--version', type=int,
                        help="Experiment version integer")

    args = parser.parse_args()
    
    log_dir = args.log_dir
    name = args.name
    version = args.version

    exp_dir = os.path.join(
        log_dir, 
        name, 
        'version_{}'.format(version))
    
    ckpt_dir = os.path.join(
        exp_dir,
        'checkpoints')
    
    # load hyperparameter/config yaml
    cfg_path = glob.glob(exp_dir+"/*cfg.yaml")[0]
    
    with open(cfg_path) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    hparams = flatten_dict(hparams)
    hparams = Namespace(**hparams)
    
    # pick the checkpoint with highest validation accuracy
    ckpts = os.listdir(ckpt_dir)
    ckpts = [x for x in os.listdir(ckpt_dir) if not x.endswith('.txt') and not x.endswith('.npz')]
    ckpt_accs = [(i, float(x.split('.ckpt')[0][-4:])) for i, x in enumerate(ckpts) if 'epoch' in x]
    ckpt_accs.sort(key = lambda x:x[1])
    idx = ckpt_accs[-1][0]

    ckpt = ckpts[idx]
    
    # load the model
    trainer = Trainer(gpus=args.gpus)

    print("TESTING: {}".format(ckpt))
    ckpt_path = os.path.join(ckpt_dir, ckpt)

    model = getattr(lowshot_models, hparams.model_type)(hparams)
    model = model.load_from_checkpoint(ckpt_path)
    
    output_file_root = os.path.join(ckpt_dir, ckpt.replace('.ckpt', ''))
    
    # run testing according to n_ls_test_ways and n_ls_test_shots
    for n_way in hparams.n_ls_test_ways:
        for n_shot in hparams.n_ls_test_shots:
            test_hparams = {'output_file_root':output_file_root,
                            'n_ways':n_way,
                            'n_shots':n_shot, 
                            'n_ls_test_iters':hparams.n_ls_test_iters,
                            }
            
            update_hparams(model.hparams, test_hparams)

            trainer.test(model)

if __name__ == "__main__":
    main()
