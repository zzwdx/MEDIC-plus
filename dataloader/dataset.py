import os
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset, default_loader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def _resolve_class_to_idx(classes, class_to_idx=None, labeled=True):
    classes = sorted(classes)
    if not labeled:
        return {c: -1 for c in classes}
    if class_to_idx is not None:
        return class_to_idx
    return {c: i for i, c in enumerate(classes)}


class DomainDataset(Dataset):
    def __init__(
        self, root_dir, domain, classes,
        class_to_idx=None, domain_class_dict=None,
        labeled=True, with_domain_label=False,
        transform=None, loader=default_loader,
    ):
        if not os.path.isdir(root_dir):
            raise ValueError(f'Path "{root_dir}" does not exist.')

        self.root_dir = root_dir
        self.domains = [domain] if isinstance(domain, str) else sorted(domain)
        self.classes = sorted(classes)
        self.domain_class_dict = domain_class_dict
        self.with_domain_label = with_domain_label
        self.transform = transform
        self.loader = loader
        self.class_to_idx = _resolve_class_to_idx(self.classes, class_to_idx, labeled)

        self.samples = []
        self.domain_labels = []
        self._load()

    def _load(self):
        for i, name in enumerate(self.domains):
            path = os.path.join(self.root_dir, name)
            if not os.path.isdir(path):
                raise ValueError(f'Domain "{name}" does not exist.')

            if self.domain_class_dict is None:
                sub_class_to_idx = self.class_to_idx
            else:
                sub_class_to_idx = {c: self.class_to_idx[c] for c in self.domain_class_dict[name]}

            samples = make_dataset(path, sub_class_to_idx, IMG_EXTENSIONS)
            self.samples.extend(samples)
            if self.with_domain_label:
                self.domain_labels.extend([i] * len(samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        domain_label = self.domain_labels[index] if self.with_domain_label else -1
        return img, label, domain_label
