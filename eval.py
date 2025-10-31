import argparse
import torch
import pickle
import os
from dataloader.dataloader import get_dataloader
from model.model import MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast, ConvNet, gfnet_fast
from util.log import Logger
from train.test import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--dataset', default=None)

    parser.add_argument('--batch-size', type=int, default=128)


    parser.add_argument('--save-dir', default='/data/wxr/MEDIC-plus/save')
    parser.add_argument('--save-name', default='demo')
    
    args = parser.parse_args()

    # It can be used to replace the following code, but the editor may take it as an error.
    # locals().update(vars(args))

    # It can be replaced by the preceding code.

    gpu = args.gpu
    dataset = args.dataset
    batch_size = args.batch_size
    save_dir = args.save_dir
    save_name = args.save_name

    param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
    
    with open(param_path, 'rb') as f: 
        param = pickle.load(f)
        
    if dataset is None:
        dataset = param['dataset']    
    if dataset == 'PACS':
        root_dir = '/data/datasets/PACS'
        small_img = False
    elif dataset == 'OfficeHome':
        root_dir = ''
        small_img = False
    elif dataset == "DigitsDG":
        root_dir = ''
        small_img = True
    elif dataset == 'VLCS':
        root_dir = ''
        small_img = False
    elif dataset == 'TerraIncognita':
        root_dir = ''
        small_img = False
    elif dataset == "DomainNet":
        root_dir = ''
        small_img = False

    source_domain = sorted(param['source_domain'])
    target_domain = sorted(param['target_domain'])
    known_classes = sorted(param['known_classes'])
    unknown_classes = sorted(param['unknown_classes'])
     
    net_name = param['net_name']
    if "share_param" in param:
        share_param = param['share_param']
    else:
        share_param = False


    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    num_classes = len(known_classes)
    log_path = os.path.join('./', save_dir, 'log', save_name + '_test.txt')
    model_path = os.path.join('./', save_dir, 'model', 'val', save_name + '.tar')

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test_k, *_ = get_dataloader(root_dir=root_dir, domain=target_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    if len(unknown_classes) > 0:
        test_u, *_ = get_dataloader(root_dir=root_dir, domain=target_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=False, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)    
    else:
        test_u = None


    if share_param:
        muticlassifier = MutiClassifier_
    else:
        muticlassifier = MutiClassifier

    if net_name == 'resnet18':
        net = muticlassifier(net=resnet18_fast(), num_classes=num_classes)
    elif net_name == 'resnet50':
        net = muticlassifier(net=resnet50_fast(), num_classes=num_classes, feature_dim=2048)
    elif net_name == 'convnet':
        net = muticlassifier(net=ConvNet(), num_classes=num_classes, feature_dim=256)
    elif net_name == 'gfnet':
        net = muticlassifier(net=gfnet_fast("/data0/xiran/MEDIC-plus-vit/save/model/pretrain/gfnet-h-ti.pth"), num_classes=num_classes, feature_dim=512)

    net.load_state_dict(torch.load(model_path))

    net = net.to(device)

    logger = Logger(log_path)

    logger.log_params(
        GPU=gpu,
        SaveName=save_name,
        Dataset=dataset,
        SourceDomain=source_domain,
        TargetDomain=target_domain,
        KnownClasses=known_classes,
        UnknownClasses=unknown_classes,
        Network=net_name,
        ShareParameter=share_param
    )

    logger.log('Start Testing...')  
    

    net.eval()
    eval_all(net, None, test_k, test_u, log_path, epoch=-1, device=device)

