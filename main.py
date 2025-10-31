from config import *
import torch
import pickle
import os
from dataloader.dataloader import get_dataloader, get_domain_specific_dataloader
from model.model import MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast, ConvNet, gfnet_fast
from optimizer.optimizer import get_optimizer, get_scheduler
from loss.OVALoss import OVALoss
from train.test import *
from util.log import Logger
from train.select import *
from util.iterator import *
from train.ml import *
import types


if __name__ == '__main__':

    logger = Logger(log_path)

    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(param_path, 'wb') as f: 
        pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.log('Loading dataset...')

    num_domain = len(source_domain)
    num_classes = len(known_classes)

    domain_index_list = [i for i in range(num_domain)]
    class_index_list = [i for i in range(num_classes)]

    num_group = 10 if num_classes >= 10 else num_classes

    if dataset == 'TerraIncognita' and len(unknown_classes) == 0:
        num_group = 5
    elif dataset == 'DomainNet' and len(unknown_classes) == 0:
        num_group = 20

    group_index_list = [i for i in range(num_group)]
    classes_partition = split_classes(classes_list=known_classes, index_list=class_index_list, n=num_group)
    group_length_list = [len(g) for g in classes_partition]


    logger.log_params(
        GPU=gpu,
        SaveName=save_name,
        SaveLater=save_later,
        Dataset=dataset,
        SourceDomain=source_domain,
        TargetDomain=target_domain,
        KnownClasses=known_classes,
        UnknownClasses=unknown_classes,
        NumGroups=num_group,
        BatchSize=batch_size,
        Algorithm=algorithm,
        TaskDomain=task_d,
        TaskClass=task_c,
        TasksPerStep=task_per_step,
        SelectionMode=selection_mode,
        CrossVal=crossval,
        Network=net_name,
        NumEpoch=num_epoch,
        EvalStep=eval_step,
        OptimizeMethod=optimize_method,
        LearningRate=lr,
        MetaLearningRate=meta_lr,
        WithoutCloseSetClassifier=without_cls,
        WithoutBinaryClassifier=without_bcls,
        ShareParameter=share_param
    )

    domain_specific_loader, val_k = get_domain_specific_dataloader(
        root_dir=train_dir, 
        domain=source_domain, 
        classes=known_classes, 
        classes_partition=classes_partition, 
        batch_size=sub_batch_size, 
        small_img=small_img, 
        crossval=crossval and random_split
    )

    if crossval and val_k == None:
        val_k, *_ = get_dataloader(
            root_dir=val_dir, 
            domain=source_domain, 
            classes=known_classes, 
            batch_size=batch_size, 
            get_domain_label=False, 
            get_class_label=True, 
            instr="val", 
            small_img=small_img, 
            shuffle=False, 
            drop_last=False, 
            num_workers=4
        )

    test_k, *_ = get_dataloader(
        root_dir=test_dir, 
        domain=target_domain, 
        classes=known_classes, 
        batch_size=batch_size, 
        get_domain_label=False, 
        get_class_label=True, 
        instr="test", 
        small_img=small_img, 
        shuffle=False, 
        drop_last=False, 
        num_workers=4
    )

    if len(unknown_classes) > 0:
        test_u, *_ = get_dataloader(
            root_dir=test_dir, 
            domain=target_domain, 
            classes=unknown_classes, 
            batch_size=batch_size, 
            get_domain_label=False, 
            get_class_label=False, 
            instr="test", 
            small_img=small_img, 
            shuffle=False, 
            drop_last=False, 
            num_workers=4
        )   
    else:
        test_u = None

    logger.log('Loading models...')

    if share_param:
        muticlassifier = MutiClassifier_
    else:
        muticlassifier = MutiClassifier

    if net_name == 'resnet18':
        net = muticlassifier(net=resnet18_fast(), num_classes=num_classes)
    elif net_name == 'resnet50':
        net = muticlassifier(net=resnet50_fast(), num_classes=num_classes, feature_dim=2048)
    elif net_name == "convnet":
        net = muticlassifier(net=ConvNet(), num_classes=num_classes, feature_dim=256)
    elif net_name == 'gfnet':
        net = muticlassifier(net=gfnet_fast("/data0/xiran/MEDIC-plus-vit/save/model/pretrain/gfnet-h-ti.pth"), num_classes=num_classes, feature_dim=512)

    net = net.to(device)
    
    if optimize_method == 'SGD':
        optimizer = get_optimizer(net=net, instr=optimize_method, lr=lr, nesterov=nesterov)
        scheduler = get_scheduler(optimizer=optimizer, instr=schedule_method, step_size=int(num_epoch*0.8), gamma=0.1)
    elif optimize_method in ['Adam', 'AdamW']:
        optimizer = get_optimizer(net=net, instr=optimize_method, lr=lr)
        scheduler = types.SimpleNamespace(step=lambda: 0)

    if num_epoch_before != 0:
        logger.log('Loading state dict...')  
        net.load_state_dict(torch.load(model_val_path))
        for epoch in range(num_epoch_before):
            scheduler.step()
        logger.log_params(
            NumEpochBefore=num_epoch_before
        )


    logger.log('Start training...')  

    recall = {
        'va': 0,
        'ta': 0,
        'oscrc': 0, 
        'oscrb': 0,
        'bva': 0, 
        'bvta': 0, 
        'bvt': [],
        'bta': 0, 
        'btt': []
    }

    criterion = torch.nn.CrossEntropyLoss()
    if without_cls:
        criterion = lambda *args: 0
    ovaloss = OVALoss()
    if without_bcls:
        ovaloss = lambda *args: 0


    task_pool = get_task_pool(task_d=task_d, task_c=task_c, domain_index_list=domain_index_list, group_index_list=group_index_list, group_length_list=group_length_list, net=net, domain_specific_loader=domain_specific_loader, device=device, mode=selection_mode)
   

    fast_parameters = list(net.parameters())
    load_fast_weights(net, None)
    net.zero_grad()

    
    for epoch in range(num_epoch_before, num_epoch):

        net.train()
        task_count = 0
        step_index = 0
        input_sum = []
        label_sum = []

        for domain_index, group_index in task_pool:
        
            for i in domain_index:
                domain_specific_loader[i].keep(group_index)
                input, label = domain_specific_loader[i].next(batch_size=batch_size//len(domain_index))
                domain_specific_loader[i].reset()

                input = input.to(device)
                label = label.to(device)
                input_sum.append(input)
                label_sum.append(label)
                
            task_count = (task_count + 1) % task_per_step[step_index]
            if task_count == 0:

                input_sum = torch.cat(input_sum, dim=0)
                label_sum = torch.cat(label_sum, dim=0)

                grad = compute_gradient(net=net, fast_parameters=fast_parameters, input=input_sum, label=label_sum, criterion=criterion, ovaloss=ovaloss, weight=task_per_step[step_index])
                fast_parameters = update_fast_weights("reptile", net=net, grad=grad, meta_lr=meta_lr)

                if algorithm == 'medic':
                    pass
                    
                elif algorithm == 'arith':
                    accumulate_meta_grads("arith", net=net, grad=grad, meta_lr=meta_lr, eta=weight_per_step[step_index])


                input_sum = []
                label_sum = []
                step_index += 1

        if algorithm == 'medic':
            accumulate_meta_grads("reptile", net=net, meta_lr=meta_lr)
            
        elif algorithm == 'arith':
            pass


        # update with original optimizers
        optimizer.step()  
    

        fast_parameters = list(net.parameters())
        load_fast_weights(net, None)
        net.zero_grad()

        task_pool = get_task_pool(task_d=task_d, task_c=task_c, domain_index_list=domain_index_list, group_index_list=group_index_list, group_length_list=group_length_list, net=net, domain_specific_loader=domain_specific_loader, device=device, mode=selection_mode) 

        if (epoch+1) % eval_step == 0:      
       
            net.eval()  
            recall['va'], recall['ta'], recall['oscrc'], recall['oscrb'] = eval_all(net, val_k, test_k, test_u, log_path, epoch, device)
            update_recall(net, recall, log_path, model_val_path)
            
        if epoch+1 == renovate_step:
                logger.log("Reset accuracy history...")
                recall['bva'] = 0
                recall['bvta'] = 0
                recall['bvt'] = []
                recall['bta'] = 0
                recall['btt'] = []

        scheduler.step()