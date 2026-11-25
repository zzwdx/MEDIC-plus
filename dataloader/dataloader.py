from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataloader.dataset import DomainDataset
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


def _make_multi_domain_dataset(root_dir, domain, classes, transform, labeled=True):
    domain = domain if isinstance(domain, list) else [domain]
    return DomainDataset(
        root_dir=root_dir, domain=domain, classes=classes,
        labeled=labeled, transform=transform
    )


def get_dataloader(
    root_dir, domain, classes, batch_size,
    instr="train", small_img=False, shuffle=True, drop_last=True,
    num_workers=4, split_rate=0.8, crossval=False, labeled=True
):
    transform = get_transform(instr, small_img=small_img)

    if isinstance(root_dir, list):
        dataset = ConcatDataset([
            _make_multi_domain_dataset(path, domain, classes, transform, labeled)
            for path in root_dir
        ])
    else:
        dataset = _make_multi_domain_dataset(root_dir, domain, classes, transform, labeled)

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


def _make_class_group_dataset(root_dir, domain, cls_info, transform):
    return DomainDataset(
        root_dir=root_dir, domain=domain, classes=cls_info.keys(),
        class_to_idx=cls_info, transform=transform
    )


def get_domain_specific_dataloader(
    root_dir, domain, classes_partition,
    batch_size, small_img, split_rate=0.8,
    crossval=False, val_workers=4
):
    domain_specific_loader, val_list = [], []
    transform = get_transform("train", small_img=small_img)

    for dom in domain:
        dataloader_list = []

        for cls_info in classes_partition:
            dataset = _make_class_group_dataset(root_dir, dom, cls_info, transform)

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

    








