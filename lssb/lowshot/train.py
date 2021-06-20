from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import TensorBoardLogger

import pytorch_lightning as pl
import os
import shutil
import yaml
import lssb.lowshot.models as lowshot_models
import subprocess

def add_arguments(parser):
    parser.add_argument('--cfg', type=str, default='')
    
    return parser

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
    parser = add_arguments(parser)
    args = parser.parse_args()
    
    with open(args.cfg) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    hparams = flatten_dict(hparams)
    hparams = Namespace(**hparams)

    if hparams.use_seed==True:
        print("Training with fixed random seed")
        seed_everything(1994)
        benchmark=False
        deterministic=True
    else:
        benchmark=True
        deterministic=False


    model = getattr(lowshot_models, hparams.model_type)(hparams)
    
    log_dir = hparams.log_dir
    name = hparams.exp_name
    logger = TensorBoardLogger(log_dir, name=name)
    
    ckpt_path = os.path.join(
        log_dir, 
        name, 
        'version_{}'.format(logger.version), 
        'checkpoints',
        '{epoch}_{val_acc:.3f}'
    )

    ckpt_dir = os.path.join(
        log_dir, 
        name, 
        'version_{}'.format(logger.version)
    )
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=2,
            save_last=True,
        filepath=ckpt_path,
        verbose=True,
    )

    trainer = pl.Trainer(gpus=hparams.gpus,
        distributed_backend='dp',
        max_epochs=hparams.max_epochs,
        check_val_every_n_epoch=hparams.val_freq,
        checkpoint_callback=checkpoint_callback,
        benchmark=benchmark,
        deterministic=deterministic,
        logger=logger
    )
    
    #bookkeeping
    shutil.copy(args.cfg, ckpt_dir)
    label = subprocess.check_output(["git", "describe", "--always"]).strip()

    #logging commit version
    with open(os.path.join(ckpt_dir, 'commit.txt'), 'w') as f:
        f.write(label.decode("utf-8"))
    
    #training the model
    trainer.fit(model)
    
if __name__ == "__main__":
    main()
