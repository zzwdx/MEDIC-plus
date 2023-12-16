import torch
import random

def split_classes(classes_list, index_list, n):
    for i in range(0, len(classes_list), n):
        classes = classes_list[i:i + n]
        index = index_list[i:i+n]
        class_to_idx = dict(zip(classes, index))
        yield classes, class_to_idx


def shuffle_list(lst):
    return random.sample(lst, len(lst))


def divide_list(lst, n):
    length = len(lst)
    quotient = length // n
    remainder = length % n

    divided_list = []
    start = 0

    for i in range(n):
        size = quotient + 1 if i < remainder else quotient
        end = start + size
        divided_list.append(lst[start:end])
        start = end

    return divided_list


def get_parameters(model):
    # note : you can direct manipulate these data reference which is related to the original models
    parameters = dict(model.named_parameters())
    states = dict(model.named_buffers())
    return parameters, states
        

class ForeverDataIterator:
    """A data iterator that will never stop producing data"""
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)

    def __next__(self):
        try:
            data = next(self.iter)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
        return data

    def __len__(self):
        return len(self.data_loader)


class ConnectedDataIterator:
    def __init__(self, dataloader_list, batch_size):
        self.dataloader_list = dataloader_list
        self.batch_size = batch_size
        
        self.length = len(self.dataloader_list)
        self.iter_list = [iter(loader) for loader in self.dataloader_list]
        self.available_set = set([i for i in range(self.length)])

    def append(self, index_list):
        self.available_set = self.available_set | set(index_list)

    def keep(self, index_list):
        self.available_set = set(index_list)

    def remove(self, index_list):
        self.available_set = self.available_set - set(index_list)

    def reset(self):
        self.available_set = set([i for i in range(len(self.dataloader_list))])

    def __next__(self):
        data_sum = []
        label_sum = []
        for i in self.available_set:
            try:
                data, label, *_ = next(self.iter_list[i])
            except StopIteration:
                self.iter_list[i] = iter(self.dataloader_list[i])
                data, label, *_ = next(self.iter_list[i])
            data_sum.append(data)
            label_sum.append(label)
        
        data_sum = torch.cat(data_sum, dim=0)
        label_sum = torch.cat(label_sum, dim=0)
        
        rand_index = random.sample([i for i in range(len(data_sum))], self.batch_size)

        return data_sum[rand_index], label_sum[rand_index]

    def next(self, batch_size=None):

        if batch_size is None:
            batch_size = self.batch_size

        data_sum = []
        label_sum = []
        for i in self.available_set:
            try:
                data, label, *_ = next(self.iter_list[i])
            except StopIteration:
                self.iter_list[i] = iter(self.dataloader_list[i])
                data, label, *_ = next(self.iter_list[i])
            data_sum.append(data)
            label_sum.append(label)
        
        data_sum = torch.cat(data_sum, dim=0)
        label_sum = torch.cat(label_sum, dim=0)
        
        rand_index = random.sample([i for i in range(len(data_sum))], batch_size)

        return data_sum[rand_index], label_sum[rand_index]