# config.py
import argparse
import os
import torch


def get_args():
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
    parser.add_argument('--algorithm', default='medic')
    parser.add_argument('--task-d', type=int, default=3)
    parser.add_argument('--task-c', type=int, default=3)
    parser.add_argument('--task-per-step', nargs='+', type=int, default=[3, 3, 3])
    parser.add_argument('--weight-per-step', nargs='+', type=float, default=[1.5, 1, 0.5], help='arith only')
    parser.add_argument('--selection-mode', default='random') # random, hard

    parser.add_argument('--net-name', default='resnet50')
    parser.add_argument('--optimize-method', default="SGD")
    parser.add_argument('--schedule-method', default='StepLR')
    parser.add_argument('--num-epoch', type=int, default=6000)
    parser.add_argument('--eval-step', type=int, default=300)
    parser.add_argument('--lr', type=float, default=2e-4) # Alpha (meta-lr) has been calculated in the following code, so it is set to 1/t of the default learning rate.
    parser.add_argument('--meta-lr', type=float, default=1e-2)
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--without-cls', action='store_true')
    parser.add_argument('--without-bcls', action='store_true')
    parser.add_argument('--share-param', action='store_true')

    parser.add_argument('--save-dir', default='/data/wxr/MEDIC-plus/save')
    parser.add_argument('--save-name', default='demo')
    parser.add_argument('--save-best-test', action='store_true')
    parser.add_argument('--save-later', action='store_true')

    parser.add_argument('--num-epoch-before', type=int, default=0)
    
    return parser.parse_args()


args = get_args()

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
algorithm = args.algorithm
task_d = args.task_d
task_c = args.task_c
task_per_step = args.task_per_step
weight_per_step = args.weight_per_step
selection_mode = args.selection_mode
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
crossval = True

if dataset == 'PACS':
    train_dir = '/data/datasets/PACS'
    val_dir = '/data/datasets/PACS'
    test_dir = '/data/datasets/PACS'
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

assert task_d * task_c == sum(task_per_step)
    