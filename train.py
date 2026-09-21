import sys
import os


curPath = os.path.abspath(os.path.dirname(__file__)) 
sys.path.append(curPath)
rootPath = os.path.split(curPath)[0]			
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])
import argparse
import os
import os.path as osp
import pprint
import random
import warnings
from pathlib import Path
import numpy as np
import yaml
import torch
from torch.utils import data
from dataset.data_reader import CTDataset,MRDataset,CTDataset_aug,MRDataset_aug, t2Dataset_aug, t1Dataset, t2Dataset, t1Dataset_aug

from model.deeplabv2 import get_deeplab_v2
from domain_adaptation.config import cfg, cfg_from_file
from domain_adaptation.train_UDA import train_domain_adaptation, train_senery

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent


def resolve_path(path_value, base_dir=REPO_ROOT):
    """Resolve absolute paths as-is and repository-relative paths from REPO_ROOT."""
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    return expanded.resolve() if expanded.is_absolute() else (Path(base_dir) / expanded).resolve()


def load_config(args):
    cfg_from_file(str(resolve_path(args.cfg)))
    cfg.TRAIN.RESTORE_FROM = str(resolve_path(cfg.TRAIN.RESTORE_FROM))
    cfg.EXP_ROOT_SNAPSHOT = str(resolve_path(cfg.EXP_ROOT_SNAPSHOT))
    cfg.EXP_ROOT_LOGS = str(resolve_path(cfg.EXP_ROOT_LOGS))


def list_path(args, dataset_name, filename):
    return str(resolve_path(Path(args.datalist_root) / dataset_name / filename))


def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) training")
    parser.add_argument('--cfg', type=str, required=True,
                        help='YAML config path (absolute or relative to the repository root).')
    parser.add_argument('--dataset', choices=['mmwhs', 'pro', 'brats'], default='mmwhs',
                        help='Dataset-specific training entry point.')
    parser.add_argument('--datalist-root', type=str, default='datalist',
                        help='Directory containing MMWHS17, BraTS2018, and Pro12 list folders.')
    parser.add_argument("--random-train", action="store_true",
                        help="not fixing random seed.")
    parser.add_argument("--tensorboard", action="store_true",
                        help="visualize training loss with tensorboardX.")
    parser.add_argument("--viz-every-iter", type=int, default=None,
                        help="visualize results.")
    return parser.parse_args()
def _init_fn(worker_id):
    np.random.seed(cfg.TRAIN.RANDOM_SEED+worker_id)

def main(args):
    #LOAD ARGS
    print('Called with args')
    # print(args)

    load_config(args)

    #auto-generate exp name if not specified

    cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}/{cfg.TRAIN.DA_METHOD}_{cfg.SOURCE}2{cfg.TARGET}'
    #pth = osp.join(cfg.EP_ROOT_SNAPSHOT, cfg.EXP_NAME)

    # auto-generate snapshot path if not specified

    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT,cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    #tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR,exist_ok=True)

        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE  = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    # print('Using config:')
    # pprint.pprint(cfg)

    # Initialization
    _init_fn = None

    torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.TRAIN.RANDOM_SEED)
    np.random.seed(cfg.TRAIN.RANDOM_SEED)
    random.seed(cfg.TRAIN.RANDOM_SEED)
    # torch.backends.cudnn.deterministic = True

    # def _init_fn(worker_id):
    #     np.random.seed(cfg.TRAIN.RANDOM_SEED+worker_id)
    if os.environ.get('ADVENT_DRY_RUN','0') == '1' :
        return

    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'

    #model = GCNTransformerSegmentation(num_classes=cfg.NUM_CLASSES)

    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM, map_location=torch.device('cpu'))
        if 'DeepLab_resnet_pretrained' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)

        print('Model loaded from:{}'.format(cfg.TRAIN.RESTORE_FROM))

    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    # DataLoaders
    train_mr_data_pth = list_path(args, 'MMWHS17', 'train_mr.txt')
    train_ct_data_pth = list_path(args, 'MMWHS17', 'train_ct.txt')
    train_mr_gt_pth = list_path(args, 'MMWHS17', 'train_mr_gt.txt')
    train_ct_gt_pth = list_path(args, 'MMWHS17', 'train_ct_gt.txt')
    val_mr_data_pth = list_path(args, 'MMWHS17', 'val_mr.txt')
    val_ct_data_pth = list_path(args, 'MMWHS17', 'val_ct.txt')
    val_mr_gt_pth = list_path(args, 'MMWHS17', 'val_mr_gt.txt')
    val_ct_gt_pth = list_path(args, 'MMWHS17', 'val_ct_gt.txt')


    transforms = None
    img_mean = cfg.TRAIN.IMG_MEAN
    if cfg.SOURCE == 'MR':
        mrtrain_dataset = MRDataset(data_pth=train_mr_data_pth, gt_pth=train_mr_gt_pth,
                                    img_mean=img_mean, transform=transforms)

        cttrain_dataset = CTDataset_aug(data_pth=train_ct_data_pth, gt_pth=train_ct_gt_pth,
                                    img_mean=img_mean, transform=transforms,aug_transform=True)
    elif cfg.SOURCE == 'CT':
        mrtrain_dataset = MRDataset_aug(data_pth=train_mr_data_pth, gt_pth=train_mr_gt_pth,
                                        img_mean=img_mean, transform=transforms,aug_transform=True)

        cttrain_dataset = CTDataset(data_pth=train_ct_data_pth, gt_pth=train_ct_gt_pth,
                                        img_mean=img_mean, transform=transforms)
    mrval_dataset   = MRDataset(data_pth=val_mr_data_pth, gt_pth=val_mr_gt_pth,
                              img_mean=img_mean, transform=transforms)

    ctval_dataset   = CTDataset(data_pth=val_ct_data_pth, gt_pth=val_ct_gt_pth, img_mean=img_mean,
                              transform=transforms)

    if cfg.SOURCE == 'MR':
        strain_dataset = mrtrain_dataset
        strain_loader = data.DataLoader(strain_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)
        trgtrain_dataset = cttrain_dataset
        trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          num_workers=cfg.NUM_WORKERS,
                                          shuffle=True,
                                          pin_memory=True,
                                          worker_init_fn=_init_fn)
        sval_dataset = mrval_dataset
        sval_loader = data.DataLoader(sval_dataset,
                                      batch_size=cfg.TRAIN.BATCH_SIZE,
                                      num_workers=cfg.NUM_WORKERS,
                                      shuffle=True,
                                      pin_memory=True,
                                      worker_init_fn=_init_fn)

        trgval_dataset = ctval_dataset
        trgval_loader = data.DataLoader(trgval_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)

    elif cfg.SOURCE == 'CT':

        strain_dataset = cttrain_dataset
        strain_loader = data.DataLoader(strain_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)
        trgtrain_dataset = mrtrain_dataset
        trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          num_workers=cfg.NUM_WORKERS,
                                          shuffle=True,
                                          pin_memory=True,
                                          worker_init_fn=_init_fn)
        sval_dataset = ctval_dataset
        sval_loader = data.DataLoader(sval_dataset,
                                      batch_size=cfg.TRAIN.BATCH_SIZE,
                                      num_workers=cfg.NUM_WORKERS,
                                      shuffle=True,
                                      pin_memory=True,
                                      worker_init_fn=_init_fn)

        trgval_dataset = mrval_dataset
        trgval_loader = data.DataLoader(trgval_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)

    print('dataloader finish')
    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    train_domain_adaptation(model,strain_loader,trgtrain_loader,sval_loader,cfg)

    #train_senery(model, strain_loader, sval_loader, trgtrain_loader, cfg)
def main_pro(args):
    #LOAD ARGS
    print('Called with args')
    # print(args)

    load_config(args)

    #auto-generate exp name if not specified

    cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}/{cfg.TRAIN.DA_METHOD}_{cfg.SOURCE}2{cfg.TARGET}'
    #pth = osp.join(cfg.EP_ROOT_SNAPSHOT, cfg.EXP_NAME)

    # auto-generate snapshot path if not specified

    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT,cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    #tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR,exist_ok=True)

        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE  = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    # print('Using config:')
    # pprint.pprint(cfg)

    # Initialization
    _init_fn = None

    torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.TRAIN.RANDOM_SEED)
    np.random.seed(cfg.TRAIN.RANDOM_SEED)
    random.seed(cfg.TRAIN.RANDOM_SEED)
    # torch.backends.cudnn.deterministic = True

    # def _init_fn(worker_id):
    #     np.random.seed(cfg.TRAIN.RANDOM_SEED+worker_id)
    if os.environ.get('ADVENT_DRY_RUN','0') == '1' :
        return

    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'

    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=2, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM, map_location=torch.device('cpu'))
        if 'DeepLab_resnet_pretrained' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)

        print('Model loaded from:{}'.format(cfg.TRAIN.RESTORE_FROM))

    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    # DataLoaders for brats
    train_hk_data_pth = list_path(args, 'Pro12', 'train_hk.txt')
    train_bidmc_data_pth = list_path(args, 'Pro12', 'train_bidmc.txt')
    train_hk_gt_pth = list_path(args, 'Pro12', 'train_hk_gt.txt')
    train_bidmc_gt_pth = list_path(args, 'Pro12', 'train_bidmc_gt.txt')


    transforms = None
    img_mean = cfg.TRAIN.IMG_MEAN
    #img_mean =np.array((0, 0,0), dtype=np.float32)

    if cfg.SOURCE == 'HK':
        hktrain_dataset = MRDataset(data_pth=train_hk_data_pth, gt_pth=train_hk_gt_pth,
                                    img_mean=img_mean, transform=transforms)

        bidmctrain_dataset = CTDataset_aug(data_pth=train_bidmc_data_pth, gt_pth=train_bidmc_gt_pth,
                                    img_mean=img_mean, transform=transforms, aug_transform=True)
    elif cfg.SOURCE == 'BIDMC':
        hktrain_dataset = MRDataset_aug(data_pth=train_hk_data_pth, gt_pth=train_hk_gt_pth,
                                    img_mean=img_mean, transform=transforms, aug_transform=True)

        bidmctrain_dataset = CTDataset(data_pth=train_bidmc_data_pth, gt_pth=train_bidmc_gt_pth,
                                    img_mean=img_mean, transform=transforms)


    if cfg.SOURCE == 'HK':
        strain_dataset = hktrain_dataset
        strain_loader = data.DataLoader(strain_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)
        trgtrain_dataset = bidmctrain_dataset
        trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          num_workers=cfg.NUM_WORKERS,
                                          shuffle=True,
                                          pin_memory=True,
                                          worker_init_fn=_init_fn)


    elif cfg.SOURCE == 'BIDMC':

        strain_dataset = bidmctrain_dataset
        strain_loader = data.DataLoader(strain_dataset,
                                        batch_size=cfg.TRAIN.BATCH_SIZE,
                                        num_workers=cfg.NUM_WORKERS,
                                        shuffle=True,
                                        pin_memory=True,
                                        worker_init_fn=_init_fn)
        trgtrain_dataset = hktrain_dataset
        trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          num_workers=cfg.NUM_WORKERS,
                                          shuffle=True,
                                          pin_memory=True,
                                          worker_init_fn=_init_fn)


    sval_loader=""
    print('dataloader finish')
    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    train_domain_adaptation(model,strain_loader,trgtrain_loader,sval_loader,cfg)
def main_brats(args):
    #LOAD ARGS
    print('Called with args')
    # print(args)

    load_config(args)

    #auto-generate exp name if not specified

    cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}/{cfg.TRAIN.DA_METHOD}_{cfg.SOURCE}2{cfg.TARGET}'
    #pth = osp.join(cfg.EP_ROOT_SNAPSHOT, cfg.EXP_NAME)

    # auto-generate snapshot path if not specified

    if cfg.TRAIN.SNAPSHOT_DIR == '':
        cfg.TRAIN.SNAPSHOT_DIR = osp.join(cfg.EXP_ROOT_SNAPSHOT,cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR, exist_ok=True)

    #tensorboard
    if args.tensorboard:
        if cfg.TRAIN.TENSORBOARD_LOGDIR == '':
            cfg.TRAIN.TENSORBOARD_LOGDIR = osp.join(cfg.EXP_ROOT_LOGS, 'tensorboard', cfg.EXP_NAME)
        os.makedirs(cfg.TRAIN.TENSORBOARD_LOGDIR,exist_ok=True)

        if args.viz_every_iter is not None:
            cfg.TRAIN.TENSORBOARD_VIZRATE  = args.viz_every_iter
    else:
        cfg.TRAIN.TENSORBOARD_LOGDIR = ''
    # print('Using config:')
    # pprint.pprint(cfg)

    # Initialization
    _init_fn = None

    torch.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed(cfg.TRAIN.RANDOM_SEED)
    torch.cuda.manual_seed_all(cfg.TRAIN.RANDOM_SEED)
    np.random.seed(cfg.TRAIN.RANDOM_SEED)
    random.seed(cfg.TRAIN.RANDOM_SEED)
    # torch.backends.cudnn.deterministic = True

    # def _init_fn(worker_id):
    #     np.random.seed(cfg.TRAIN.RANDOM_SEED+worker_id)
    if os.environ.get('ADVENT_DRY_RUN','0') == '1' :
        return

    assert osp.exists(cfg.TRAIN.RESTORE_FROM), f'Missing init model {cfg.TRAIN.RESTORE_FROM}'

    if cfg.TRAIN.MODEL == 'DeepLabv2':
        model = get_deeplab_v2(num_classes=2, multi_level=cfg.TRAIN.MULTI_LEVEL)
        saved_state_dict = torch.load(cfg.TRAIN.RESTORE_FROM, map_location=torch.device('cpu'))
        if 'DeepLab_resnet_pretrained' in cfg.TRAIN.RESTORE_FROM:
            new_params = model.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[1] == 'layer5':
                    new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
            model.load_state_dict(new_params)
        else:
            model.load_state_dict(saved_state_dict)

        print('Model loaded from:{}'.format(cfg.TRAIN.RESTORE_FROM))

    else:
        raise NotImplementedError(f"Not yet supported {cfg.TRAIN.MODEL}")

    # DataLoaders for brats
    if cfg.SOURCE == 't2':
        train_t2_data_pth = list_path(args, 'BraTS2018', 'brats_train_t2.txt')
        train_tar_data_pth = list_path(args, 'BraTS2018', 'brats_train_flair.txt')
        train_t2_gt_pth = list_path(args, 'BraTS2018', 'brats_train_t2_gt.txt')
        train_tar_gt_pth = list_path(args, 'BraTS2018', 'brats_train_flair_gt.txt')
    elif cfg.SOURCE == 'flair':
        train_t2_data_pth = list_path(args, 'BraTS2018', 'brats_train_flair.txt')
        train_tar_data_pth = list_path(args, 'BraTS2018', 'brats_train_t2.txt')
        train_t2_gt_pth = list_path(args, 'BraTS2018', 'brats_train_flair_gt.txt')
        train_tar_gt_pth = list_path(args, 'BraTS2018', 'brats_train_t2_gt.txt')

    val_t2_data_pth = list_path(args, 'BraTS2018', 'brats_val_t2.txt')
    val_tar_data_pth = list_path(args, 'BraTS2018', 'brats_val_flair.txt')
    val_t2_gt_pth = list_path(args, 'BraTS2018', 'brats_val_t2_gt.txt')
    val_tar_gt_pth = list_path(args, 'BraTS2018', 'brats_val_flair_gt.txt')




    transforms = None
    img_mean = cfg.TRAIN.IMG_MEAN
    #img_mean =np.array((0, 0,0), dtype=np.float32)

    t2train_dataset = t2Dataset(data_pth=train_t2_data_pth, gt_pth=train_t2_gt_pth,
                                    img_mean=img_mean, transform=transforms)

    tartrain_dataset = t1Dataset_aug(data_pth=train_tar_data_pth, gt_pth=train_tar_gt_pth,
                                    img_mean=img_mean, transform=transforms,aug_transform=True)


    t2val_dataset   = t2Dataset(data_pth=val_t2_data_pth, gt_pth=val_t2_gt_pth,
                              img_mean=img_mean, transform=transforms)

    tarval_dataset   = t1Dataset(data_pth=val_tar_data_pth, gt_pth=val_tar_gt_pth, img_mean=img_mean,
                              transform=transforms)


    strain_dataset = t2train_dataset
    strain_loader = data.DataLoader(strain_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)
    trgtrain_dataset = tartrain_dataset
    trgtrain_loader = data.DataLoader(trgtrain_dataset,
                                      batch_size=cfg.TRAIN.BATCH_SIZE,
                                      num_workers=cfg.NUM_WORKERS,
                                      shuffle=True,
                                      pin_memory=True,
                                      worker_init_fn=_init_fn)
    sval_dataset = t2val_dataset
    sval_loader = data.DataLoader(sval_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  num_workers=cfg.NUM_WORKERS,
                                  shuffle=True,
                                  pin_memory=True,
                                  worker_init_fn=_init_fn)

    trgval_dataset = tarval_dataset
    trgval_loader = data.DataLoader(trgval_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_init_fn)



    print('dataloader finish')
    with open(osp.join(cfg.TRAIN.SNAPSHOT_DIR, 'train_cfg.yml'), 'w') as yaml_file:
        yaml.dump(cfg, yaml_file, default_flow_style=False)

    # UDA TRAINING
    train_domain_adaptation(model,strain_loader,trgtrain_loader,sval_loader,cfg)

    #train_senery(model, strain_loader, sval_loader, trgtrain_loader, cfg)

if __name__ == '__main__':
    args = get_arguments()
    os.chdir(REPO_ROOT)
    if args.dataset == 'mmwhs':
        main(args)
    elif args.dataset == 'pro':
        main_pro(args)
    else:
        main_brats(args)
