from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataloader.dataset import SingleClassData, MultiClassData, MultiDomainData
from util.iterator import *


def get_transform(instr, small_img=False, color_jitter=True, random_grayscale=True):
    size = (224, 224) if not small_img else (32, 32)
    normalize_mean_std = (
        ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if not small_img
        else ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    )

    train_transforms = (
        [transforms.RandomResizedCrop(size, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(0.5)] # 0.7 for DomainBed, otherwise 0.8
        if not small_img else [transforms.Resize(size)]
    )

    if color_jitter:
        train_transforms.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)) # 0.3 for DomainBed, otherwise 0.4
    if random_grayscale:
        train_transforms.append(transforms.RandomGrayscale(0.1))

    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(*normalize_mean_std)
    ]

    eval_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(*normalize_mean_std)
    ])

    transform_map = {
        "train": transforms.Compose(train_transforms),
        "val": eval_transforms,
        "test": eval_transforms
    }

    return transform_map[instr]


def get_dataloader(
    root_dir, domain, classes, batch_size,
    domain_class_dict=None, get_domain_label=True, get_class_label=True,
    instr="train", small_img=False, shuffle=True, drop_last=True,
    num_workers=4, split_rate=0.8, crossval=False
):
    domain = domain if isinstance(domain, list) else [domain]

    transform = get_transform(instr, small_img=small_img)

    if isinstance(root_dir, list):
        datasets = [
            MultiDomainData(
                root_dir=path, domain=domain, classes=classes,
                domain_class_dict=domain_class_dict,
                get_domain_label=get_domain_label,
                get_classes_label=get_class_label,
                transform=transform
            )
            for path in root_dir
        ]
        dataset = ConcatDataset(datasets)
    else:
        dataset = MultiDomainData(
            root_dir=root_dir, domain=domain, classes=classes,
            domain_class_dict=domain_class_dict,
            get_domain_label=get_domain_label,
            get_classes_label=get_class_label,
            transform=transform
        )

    val_loader = None
    if crossval:
        train_size = int(len(dataset) * split_rate)
        val_size = len(dataset) - train_size
        dataset, val_dataset = random_split(dataset, [train_size, val_size])

        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            drop_last=False, num_workers=num_workers
        )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, num_workers=num_workers
    )

    return dataloader, val_loader


def get_domain_specific_dataloader(
    root_dir, domain, classes, classes_partition,
    batch_size, small_img, split_rate=0.8,
    crossval=False, val_workers=4
):
    domain_specific_loader, val_list = [], []

    is_single_class = len(classes_partition) == len(classes)
    transform = get_transform("train", small_img=small_img)

    for dom in domain:
        dataloader_list = []

        for i, cls_info in enumerate(classes) if is_single_class else enumerate(classes_partition):
            if is_single_class:
                dataset = SingleClassData(
                    root_dir=root_dir, domain=dom, classes=cls_info,
                    domain_label=-1, classes_label=i, transform=transform
                )
            else:
                dataset = MultiClassData(
                    root_dir=root_dir, domain=dom, classes=cls_info.keys(),
                    domain_label=-1, get_classes_label=True,
                    class_to_idx=cls_info, transform=transform
                )

            if crossval:
                train_size = int(len(dataset) * split_rate)
                dataset, val = random_split(dataset, [train_size, len(dataset) - train_size])
                val_list.append(val)

            dataloader_list.append(DataLoader(
                dataset=dataset, batch_size=batch_size,
                shuffle=True, drop_last=True, num_workers=1
            ))

        domain_specific_loader.append(
            ConnectedDataIterator(dataloader_list=dataloader_list, batch_size=batch_size)
        )

    val_loader = DataLoader(
        dataset=ConcatDataset(val_list),
        batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=val_workers
    ) if crossval else None

    return domain_specific_loader, val_loader

    








