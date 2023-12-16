import argparse
import torch
import pickle
import os
from dataset.dataloader import get_dataloader, get_domain_specific_dataloader
from model.model import MutiClassifier, MutiClassifier_, resnet18_fast, resnet50_fast, ConvNet
from optimizer.optimizer import get_optimizer, get_scheduler
from loss.OVALoss import OVALoss
from train.test import *
from util.log import log
from util.util import *
import types

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', default='PACS')
    parser.add_argument('--source-domain', nargs='+', default=['photo', 'cartoon', 'art_painting'])
    parser.add_argument('--target-domain', nargs='+', default=['sketch'])
    parser.add_argument('--known-classes', nargs='+', default=['dog', 'elephant', 'giraffe', 'horse', 'guitar', 'house', 'person'])
    parser.add_argument('--unknown-classes', nargs='+', default=[])
    
    # parser.add_argument('--dataset', default='OfficeHome')
    # parser.add_argument('--source-domain', nargs='+', default=['Art', 'Clipart', 'Product'])
    # parser.add_argument('--target-domain', nargs='+', default=['RealWorld'])
    # parser.add_argument('--known-classes', nargs='+', default=['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 
    #     'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 
    #     'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 
    #     'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 
    #     'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 
    #     'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard',  
    #     'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 
    #     'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven',
    #     'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 
    #     'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler',
    #     'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 
    #     'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 
    #     'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam'
    #     ])
    # parser.add_argument('--unknown-classes', nargs='+', default=[ 
            

    #     ])

    # parser.add_argument('--dataset', default='DigitsDG')
    # parser.add_argument('--source-domain', nargs='+', default=['mnist', 'mnist_m', 'svhn'])
    # parser.add_argument('--target-domain', nargs='+', default=['syn'])
    # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4', '5'])
    # parser.add_argument('--unknown-classes', nargs='+', default=['6', '7', '8', '9'])

    # parser.add_argument('--dataset', default='VLCS')
    # parser.add_argument('--source-domain', nargs='+', default=['CALTECH', 'PASCAL', 'SUN'])
    # parser.add_argument('--target-domain', nargs='+', default=['LABELME',])
    # parser.add_argument('--known-classes', nargs='+', default=['0', '1', '2', '3', '4'])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    # parser.add_argument('--dataset', default='TerraIncognita')
    # parser.add_argument('--source-domain', nargs='+', default=['location_38', 'location_43', 'location_46'])
    # parser.add_argument('--target-domain', nargs='+', default=['location_100'])
    # parser.add_argument('--known-classes', nargs='+', default=['bobcat', 'coyote', 'dog', 'opossum', 'rabbit', 'raccoon', 'squirrel', 'bird', 'cat', 'empty',])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    # parser.add_argument('--dataset', default='DomainNet')
    # parser.add_argument('--source-domain', nargs='+', default=['clipart', 'infograph', 'painting', 'quickdraw', 'real'])
    # parser.add_argument('--target-domain', nargs='+', default=['sketch'])
    # parser.add_argument('--known-classes', nargs='+', default=['aircraft_carrier', 'airplane', 'alarm_clock', 'ambulance', 'angel', 
    #     'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 
    #     'barn', 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 
    #     'bed', 'bee', 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 
    #     'book', 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom', 
    #     'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel', 
    #     'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 
    #     'ceiling_fan', 'cello', 'cell_phone', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 
    #     'coffee_cup', 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 
    #     'crown', 'cruise_ship', 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 
    #     'dragon', 'dresser', 'drill', 'drums', 'duck', 
    #     'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses', 'face', 'fan', 
    #     'feather', 'fence', 'finger', 'fireplace', 'firetruck', 'fire_hydrant', 'fish', 'flamingo', 'flashlight', 'flip_flops', 
    #     'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan', 'garden', 'garden_hose', 'giraffe',
    #     'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger', 'hammer', 'hand', 'harp', 'hat', 
    #     'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck', 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 
    #     'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant', 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 
    #     'key', 'keyboard', 'knee', 'knife', 'ladder', 'lantern', 'laptop', 'leaf', 'leg', 'lighter', 
    #     'lighthouse', 'lightning', 'light_bulb', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox', 'map', 
    #     'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito', 'motorbike', 
    #     'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean', 
    #     'octagon', 'octopus', 'onion', 'oven', 'owl', 'paintbrush', 'paint_can', 'palm_tree', 'panda', 'pants', 
    #     'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano', 
    #     'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond', 'pool', 
    #     'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain', 'rainbow', 
    #     'rake', 'remote_control', 'rhinoceros', 'rifle', 'river', 'rollerskates', 'roller_coaster', 'sailboat', 'sandwich', 'saw', 
    #     'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw', 'shark', 'sheep', 'shoe', 
    #     'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag', 'smiley_face', 'snail', 'snake', 
    #     'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat', 'spider', 'spoon', 'spreadsheet', 'square', 
    #     'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo', 'stethoscope', 'stitches', 'stop_sign', 'stove', 
    #     'strawberry', 'streetlight', 'string_bean', 'submarine', 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 
    #     'syringe', 't-shirt', 'table', 'teapot', 'teddy-bear', 'telephone', 'television', 'tennis_racquet', 'tent', 'The_Eiffel_Tower', 
    #     'The_Great_Wall_of_China', 'The_Mona_Lisa', 'tiger', 'toaster', 'toe', 'toilet', 'tooth', 'toothbrush', 'toothpaste', 'tornado', 
    #     'tractor', 'traffic_light', 'train', 'tree', 'triangle', 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 
    #     'van', 'vase', 'violin', 'washing_machine', 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 
    #     'wine_glass', 'wristwatch', 'yoga', 'zebra', 'zigzag'
    #     ])
    # parser.add_argument('--unknown-classes', nargs='+', default=[])

    parser.add_argument('--random-split', action='store_true')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--task-d', type=int, default=3)
    parser.add_argument('--task-c', type=int, default=3)
    parser.add_argument('--task-per-step', nargs='+', type=int, default=[3, 3, 3])

    parser.add_argument('--net-name', default='resnet50')
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=6000)
    parser.add_argument('--eval-step', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-4) 
    parser.add_argument('--meta-lr', type=float, default=1e-2)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--without-cls', action='store_true')
    parser.add_argument('--without-bcls', action='store_true')
    parser.add_argument('--share-param', action='store_true')

    parser.add_argument('--save-dir', default='/data/wxr/project/ML-osdg/save')
    parser.add_argument('--save-name', default='demo')
    parser.add_argument('--save-best-test', action='store_true')
    parser.add_argument('--save-later', action='store_true')

    parser.add_argument('--num-epoch-before', type=int, default=0)
    
    args = parser.parse_args()

    # It can be used to replace the following code, but the editor may take it as an error.
    # locals().update(vars(args))

    # It can be replaced by the preceding code.
    dataset = args.dataset
    source_domain = sorted(args.source_domain)
    target_domain = sorted(args.target_domain)
    known_classes = sorted(args.known_classes)
    unknown_classes = sorted(args.unknown_classes)   
    random_split = args.random_split
    gpu = args.gpu
    batch_size = args.batch_size
    task_d = args.task_d
    task_c = args.task_c
    task_per_step = args.task_per_step
    net_name = args.net_name
    optimize_method = args.optimize_method
    schedule_method = args.schedule_method
    num_epoch = args.num_epoch
    eval_step = args.eval_step
    lr = args.lr
    meta_lr = args.meta_lr
    nesterov = args.nesterov
    without_cls = args.without_cls
    without_bcls = args.without_bcls
    share_param = args.share_param
    save_dir = args.save_dir
    save_name = args.save_name   
    save_later = args.save_later
    save_best_test = args.save_best_test
    num_epoch_before = args.num_epoch_before

    assert task_d * task_c == sum(task_per_step)

    torch.set_num_threads(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    crossval = True

    if dataset == 'PACS':
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size // 2    
        small_img = False
    elif dataset == 'OfficeHome':
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size // 2
        small_img = False
    elif dataset == "DigitsDG":
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size // 2
        small_img = True
    elif dataset == 'VLCS':
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size 
        small_img = False
    elif dataset == 'TerraIncognita':
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size
        small_img = False
    elif dataset == "DomainNet":
        train_dir = ''
        val_dir = ''
        test_dir = ''
        sub_batch_size = batch_size // 2
        small_img = False
    
    
    log_path = os.path.join(save_dir, 'log', save_name + '_train.txt')
    param_path = os.path.join(save_dir, 'param', save_name + '.pkl')
    model_val_path = os.path.join(save_dir, 'model', 'val', save_name + '.tar')
    model_test_path = os.path.join(save_dir, 'model', 'test', save_name + '.tar')
    renovate_step = int(num_epoch*0.85) if save_later else 0

    log('GPU: {}'.format(gpu), log_path)

    log('Loading path...', log_path)

    log('Save name: {}'.format(save_name), log_path)
    log('Save best test: {}'.format(save_best_test), log_path)
    log('Save later: {}'.format(save_later), log_path)

    with open(param_path, 'wb') as f: 
        pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)

    log('Loading dataset...', log_path)

    num_domain = len(source_domain)
    num_classes = len(known_classes)

    class_index = [i for i in range(num_classes)]
    group_length = (num_classes-1) // 10 + 1

    if dataset == "OfficeHome" and len(unknown_classes) == 0:
        group_length = 6
    elif dataset == 'TerraIncognita' and len(unknown_classes) == 0:
        group_length = 2
    elif dataset == 'DomainNet' and len(unknown_classes) == 0:
        group_length = 35

    log('Group length: {}'.format(group_length), log_path)
    
    group_index_list = [i for i in range((num_classes-1)//group_length + 1)]
    num_group = len(group_index_list)
 
    domain_specific_loader, val_k = get_domain_specific_dataloader(root_dir = train_dir, domain=source_domain, classes=known_classes, group_length=group_length, batch_size=sub_batch_size, small_img=small_img, crossval=crossval and random_split)
    if crossval and val_k == None:
        val_k, *_ = get_dataloader(root_dir=val_dir, domain=source_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="val", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)

 
    test_k, *_ = get_dataloader(root_dir=test_dir, domain=target_domain, classes=known_classes, batch_size=batch_size, get_domain_label=False, get_class_label=True, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)
    if len(unknown_classes) > 0:
        test_u, *_ = get_dataloader(root_dir=test_dir, domain=target_domain, classes=unknown_classes, batch_size=batch_size, get_domain_label=False, get_class_label=False, instr="test", small_img=small_img, shuffle=False, drop_last=False, num_workers=4)   
    else:
        test_u = None

    log('DataSet: {}'.format(dataset), log_path)
    log('Source domain: {}'.format(source_domain), log_path)
    log('Target domain: {}'.format(target_domain), log_path)
    log('Known classes: {}'.format(known_classes), log_path)
    log('Unknown classes: {}'.format(unknown_classes), log_path)
    log('Batch size: {}'.format(batch_size), log_path)
    log('Number of task(domain): {}'.format(task_d), log_path)
    log('Number of task(class): {}'.format(task_c), log_path)
    log('Tasks per step: {}'.format(task_per_step), log_path)
    log('CrossVal: {}'.format(crossval), log_path)
    log('Loading models...', log_path)

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


    net = net.to(device)
    
    if optimize_method == 'SGD':
        optimizer = get_optimizer(net=net, instr=optimize_method, lr=lr, nesterov=nesterov)
        scheduler = get_scheduler(optimizer=optimizer, instr=schedule_method, step_size=int(num_epoch*0.8), gamma=0.1)
    elif optimize_method == 'Adam':
        optimizer = get_optimizer(net=net, instr=optimize_method, lr=lr)
        scheduler = types.SimpleNamespace(step=lambda: 0)

    log('Network: {}'.format(net_name), log_path)
    log('Number of epoch: {}'.format(num_epoch), log_path)
    log('Optimize method: {}'.format(optimize_method), log_path)
    log('Learning rate: {}'.format(lr), log_path)
    log('Meta learning rate: {}'.format(meta_lr), log_path)

    if num_epoch_before != 0:
        log('Loading state dict...', log_path)  
        if save_best_test == False:
            net.load_state_dict(torch.load(model_val_path))
        else:
            net.load_state_dict(torch.load(model_test_path))
        for epoch in range(num_epoch_before):
            scheduler.step()
        log('Number of epoch-before: {}'.format(num_epoch_before), log_path)

    log('Without close set classifier: {}'.format(without_cls), log_path)
    log('Without binary classifier: {}'.format(without_bcls), log_path)
    log('Share Parameter: {}'.format(share_param), log_path)

    log('Start training...', log_path)  

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


    domain_index_list = [i for i in range(num_domain)]
    domain_split = divide_list(shuffle_list(domain_index_list), task_d)
    group_split = divide_list(shuffle_list(group_index_list), task_c)
    task_pool = shuffle_list([(id, ig) for id in range(task_d) for ig in range(task_c)])
   

    fast_parameters = list(net.parameters())
    for weight in net.parameters():
        weight.fast = None
    net.zero_grad()
    
    for epoch in range(num_epoch_before, num_epoch):

        net.train()
        task_count = 0
        step_index = 0
        input_sum = []
        label_sum = []

        for id, ig in task_pool:
            domain_index = domain_split[id]
            group_index = group_split[ig]
        
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

                out_c, out_b = net.c_forward(x=input_sum)
                out_b = out_b.view(out_b.size(0), 2, -1)
                loss = criterion(out_c, label_sum) + ovaloss(out_b, label_sum)
                loss *= task_per_step[step_index]
                
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=False, allow_unused=True)
                grad = [g.detach() if g is not None else g for g in grad] 

                fast_parameters = []
                for k, weight in enumerate(net.parameters()):
                    if grad[k] is not None:
                        if weight.fast is None:
                            weight.fast = weight - meta_lr * grad[k]
                        else:
                            weight.fast = weight.fast - meta_lr * grad[k]    

                    if weight.fast is None:
                        fast_parameters.append(weight)
                    else:
                        fast_parameters.append(weight.fast)
                    
                    
                input_sum = []
                label_sum = []
                step_index += 1

        optimizer.zero_grad()

        for k, weight in enumerate(net.parameters()):
            if weight.fast is None:
                weight.grad = None
            else:
                weight.grad = (weight - weight.fast) / meta_lr   

        # update original optimizers
        optimizer.step()  

        domain_split = divide_list(shuffle_list(domain_index_list), task_d)
        group_split = divide_list(shuffle_list(group_index_list), task_c) 
        task_pool = shuffle_list([(id, ig) for id in range(task_d) for ig in range(task_c)]) 

        fast_parameters = list(net.parameters())
        for weight in net.parameters():
            weight.fast = None
        net.zero_grad()

        if (epoch+1) % eval_step == 0:      
       
            net.eval()  

            recall['va'], recall['ta'], recall['oscrc'], recall['oscrb'] = eval_all(net, val_k, test_k, test_u, log_path, epoch, device)
            update_recall(net, recall, log_path, model_val_path)

            
        if epoch+1 == renovate_step:
                log("Reset accuracy history...", log_path)
                recall['bva'] = 0
                recall['bvta'] = 0
                recall['bvt'] = []
                recall['bta'] = 0
                recall['btt'] = []

        scheduler.step()