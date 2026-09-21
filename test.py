import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__)) 
sys.path.append(curPath)
rootPath = os.path.split(curPath)[0]				
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])
import argparse
import scipy.io as scio
import warnings
from pathlib import Path

from domain_adaptation.eval_UDA import eval, eval_during_train, load_checkpoint_for_evaluation
from model.deeplabv2 import get_deeplab_v2
from domain_adaptation.config import cfg, cfg_from_file
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore")
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TEST_LISTS = {
    'CT': 'datalist/MMWHS17/test_ct.txt',
    'MR': 'datalist/MMWHS17/test_mr.txt',
    't2': 'datalist/BraTS2018/brats_test_t2.txt',
    'flair': 'datalist/BraTS2018/brats_test_flair.txt',
    'hk': 'datalist/Pro12/hk_test.txt',
    'bidmc': 'datalist/Pro12/bidmc_test.txt',
}


def resolve_path(path_value, base_dir=REPO_ROOT):
    expanded = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    return expanded.resolve() if expanded.is_absolute() else (Path(base_dir) / expanded).resolve()


def read_path_list(list_path, data_root):
    with open(resolve_path(list_path), encoding='utf-8') as fp:
        return [str(resolve_path(row.strip(), data_root)) for row in fp if row.strip()]


def get_arguments():
    """
    Parse input arguments
    """

    parser = argparse.ArgumentParser(description="Code for domain adaptation (DA) evaluation")
    parser.add_argument('--cfg', type=str, required=True,
                        help='YAML config path (absolute or relative to the repository root).')
    parser.add_argument('--pretrained_model_pth', type=str, required=True,
                        help='Checkpoint path (absolute or relative to the repository root).')
    parser.add_argument('--test-list', type=str, default=None,
                        help='Optional test-list override; defaults to the released list for the modality.')
    parser.add_argument('--data-root', type=str, default='.',
                        help='Base directory for relative entries inside the test-list file.')
    parser.add_argument('--target_modality', type=str, default='MR',
                        help='optional modality', )
    parser.add_argument('--num_class', type=int, default=5, help='number of classes',)
    parser.add_argument('--dataset', type=str, default='mmwhs',
                        help='optional dataset', )
    parser.add_argument('--Method', type=str, default='test',
                        help='optional method', )

    return parser.parse_args()


def main():
    #LOAD ARGS
    args = get_arguments()

    cfg_from_file(str(resolve_path(args.cfg)))
    target_modality = args.target_modality
    if args.test_list is None and target_modality not in DEFAULT_TEST_LISTS:
        raise ValueError('Unknown target modality; provide --test-list explicitly.')
    test_list_pth = args.test_list or DEFAULT_TEST_LISTS[target_modality]
    testfile_list = read_path_list(test_list_pth, resolve_path(args.data_root))

    model = None
    if cfg.TRAIN.MODEL == 'DeepLabv2':

        model = get_deeplab_v2(num_classes = args.num_class,multi_level=cfg.TRAIN.MULTI_LEVEL)


    pretrained_model_pth = str(resolve_path(args.pretrained_model_pth))
    load_checkpoint_for_evaluation(model, pretrained_model_pth)
    Method = args.Method
    print('target_modality is {},method is {}'.format(target_modality,args.Method))
    #dice_mean,dice_std,assd_mean,assd_std,ece_value = eval(model,testfile_list,target_modality,pretrained_model_pth,Method, save_img=True)
    dice_mean, dice_std, assd_mean, assd_std= eval(model, testfile_list, target_modality,
                                                               pretrained_model_pth, Method, save_img=True,dataset=args.dataset)


    print(dice_mean.mean(), dice_std.mean(), assd_mean.mean(), assd_std.mean())

if __name__ == '__main__':
    main()
