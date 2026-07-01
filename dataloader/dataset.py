import os
import os.path
from torch.utils.data import Dataset
from torchvision.datasets.folder import make_dataset, default_loader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class SingleClassData(Dataset):
    def __init__(self, root_dir, domain, classes, domain_label=-1, classes_label=-1, transform=None, loader=default_loader):          
        
        if not os.path.isdir(root_dir):
            raise ValueError("Path \"{}\" does not exit.".format(root_dir))  

        super().__init__()
        self.root_dir = root_dir
        self.domain_name = domain
        self.class_name = classes
        self.domain_label = domain_label
        self.classes_label = classes_label
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.length = 0
        
        self.load_dataset()

    def load_dataset(self):
        class_to_idx = {self.class_name: self.classes_label}      
        path = os.path.join(self.root_dir, self.domain_name)

        if not os.path.isdir(path):
            raise ValueError("Domain \"{}\" does not exit.".format(self.domain_name))  

        self.samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
        self.length = len(self.samples)

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.domain_label


class MultiClassData(Dataset):
    def __init__(self, root_dir, domain, classes, domain_label=-1, get_classes_label=True, class_to_idx=None, transform=None, loader=default_loader):          
        
        if not os.path.isdir(root_dir):
            raise ValueError("Path \"{}\" does not exit.".format(root_dir))  

        super().__init__()
        self.root_dir = root_dir
        self.domain_name = domain
        self.classes = sorted(classes)
        self.class_to_idx = class_to_idx 
        self.domain_label = domain_label
        self.get_classes_label = get_classes_label
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.length = 0
        
        self.load_dataset()

    def load_dataset(self):
        if self.get_classes_label == False:
            class_to_idx = {self.classes[i]: -1 for i in range(len(self.classes))}           
        elif self.class_to_idx is None:
            class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        else:
            class_to_idx = self.class_to_idx
            
        path = os.path.join(self.root_dir, self.domain_name)

        if not os.path.isdir(path):
            raise ValueError("Domain \"{}\" does not exit.".format(self.domain_name))  

        self.samples = make_dataset(path, class_to_idx, IMG_EXTENSIONS)
        self.length = len(self.samples)

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.domain_label


class MultiDomainData(Dataset):
    def __init__(self, root_dir, domain, classes, domain_class_dict=None, get_domain_label=False, get_classes_label=True, transform=None, loader=default_loader):          
        
        if not os.path.isdir(root_dir):
            raise ValueError("Path \"{}\" does not exit.".format(root_dir))  

        super().__init__()
        self.root_dir = root_dir
        self.domain = sorted(domain)
        self.classes = sorted(classes)
        self.domain_class_dict = domain_class_dict
        self.get_domain_label = get_domain_label
        self.get_classes_label = get_classes_label
        self.transform = transform
        self.loader = loader
        self.samples = []
        self.domain_label = []

        self.load_dataset()

    def load_dataset(self):
        if self.get_classes_label:
            class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        else:
            class_to_idx = {self.classes[i]: -1 for i in range(len(self.classes))}
        
        for i, domain_name in enumerate(self.domain): 
            path = os.path.join(self.root_dir, domain_name)

            if not os.path.isdir(path):
                raise ValueError("Domain \"{}\" does not exit.".format(domain_name))  

            if self.domain_class_dict is None:
                sub_class_to_idx = class_to_idx
            else:
                sub_class_to_idx = {the_class: class_to_idx[the_class] for the_class in self.domain_class_dict[domain_name]}
            
            samples = make_dataset(path, sub_class_to_idx, IMG_EXTENSIONS)
            self.samples.extend(samples)
            if self.get_domain_label: 
                domain_label = [i] * len(samples)
                self.domain_label.extend(domain_label)
            

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        target = self.domain_label[index] if self.get_domain_label else -1

        return img, label, target

