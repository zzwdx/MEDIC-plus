from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataset.dataset import SingleClassData, SingleDomainData, MultiDomainData
from util.util import *

def get_transform(instr, small_img=False, color_jitter=True, random_grayscale=True):
    if small_img == False:
        img_tr = [transforms.RandomResizedCrop((224, 224), (0.8, 1.0))] # 0.7 for DomainBed, otherwise 0.8
        img_tr.append(transforms.RandomHorizontalFlip(0.5))
    else: 
        img_tr = [transforms.Resize((32, 32))]
          
    if color_jitter:
        img_tr.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)) # 0.3 for DomainBed, otherwise 0.4
    if random_grayscale:
        img_tr.append(transforms.RandomGrayscale(0.1))
    img_tr.append(transforms.ToTensor())

    if small_img == False:
        img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform_map = {     
            "train": transforms.Compose(img_tr),
            "val": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    else:       
        img_tr.append(transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        transform_map = {     
            "train": transforms.Compose(img_tr),
            "val": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            "test": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }

    return transform_map[instr]


def get_dataloader(root_dir, domain, classes, batch_size, domain_class_dict=None, get_domain_label=True, get_class_label=True, instr="train", small_img=False, shuffle=True, drop_last=True, num_workers=4, split_rate=0.8, crossval=False):
    if not isinstance(domain, list): 
        domain = [domain]

    if isinstance(root_dir, list): 
        dataset_list = []
        for path in root_dir:
            sub_dataset = MultiDomainData(root_dir=path, domain=domain, classes=classes, domain_class_dict=domain_class_dict, get_domain_label=get_domain_label, get_classes_label=get_class_label, transform=get_transform(instr, small_img=small_img))
            dataset_list.append(sub_dataset)
        dataset = ConcatDataset(dataset_list)
    else:    
        dataset = MultiDomainData(root_dir=root_dir, domain=domain, classes=classes, domain_class_dict=domain_class_dict, get_domain_label=get_domain_label, get_classes_label=get_class_label, transform=get_transform(instr, small_img=small_img))


    val_loader = None
    if crossval: 
        train_size = int(len(dataset)*split_rate)
        val_size = len(dataset) - train_size
        dataset, val = random_split(dataset, [train_size, val_size])
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    
    return dataloader, val_loader


def get_domain_specific_dataloader(root_dir, domain, classes, group_length, batch_size, small_img, split_rate=0.8, crossval=False, val_workers=4):
    domain_specific_loader = []
    val_list = [] 

    for domain_name in domain: 
        dataloader_list = []
        if group_length == 1:
            for i, class_name in enumerate(classes):
                dataset = SingleClassData(root_dir=root_dir, domain=domain_name, classes=class_name, domain_label=-1, classes_label=i, transform=get_transform("train", small_img=small_img))

                if crossval:
                    train_size = int(len(dataset)*split_rate)
                    val_size = len(dataset) - train_size
                    scd, val = random_split(dataset, [train_size, val_size])
                    val_list.append(val)
                else:
                    scd = dataset

                loader = DataLoader(dataset=scd, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
                dataloader_list.append(loader)
        else:
            classes_partition = split_classes(classes_list=classes, index_list=[i for i in range(len(classes))], n=group_length)
            for class_name, class_to_idx in classes_partition:
                dataset = SingleDomainData(root_dir=root_dir, domain=domain_name, classes=class_name, domain_label=-1, get_classes_label=True, class_to_idx=class_to_idx, transform=get_transform("train", small_img=small_img))

                if crossval:
                    train_size = int(len(dataset)*split_rate)
                    val_size = len(dataset) - train_size
                    sgd, val = random_split(dataset, [train_size, val_size])
                    val_list.append(val)
                else:
                    sgd = dataset

                loader = DataLoader(dataset=sgd, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)
                dataloader_list.append(loader)

        domain_specific_loader.append(ConnectedDataIterator(dataloader_list=dataloader_list, batch_size=batch_size))

    val_loader = None
    if crossval:
        val_loader = DataLoader(dataset=ConcatDataset(val_list), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=val_workers)

    
    return domain_specific_loader, val_loader

    








