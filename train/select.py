import torch
import torch.nn.functional as F
import random
import copy


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


def split_classes(classes_list, index_list, n):
    new_classes_list = divide_list(classes_list, n)
    new_index_list = divide_list(index_list, n)

    classes_partition = []
    for i in range(n):
        classes_partition.append(dict(zip(new_classes_list[i], new_index_list[i])))

    return classes_partition


def compute_transition_matrix(logits: torch.Tensor, labels: torch.Tensor):
    
    n_classes = logits.size(1)
    probs = F.softmax(logits, dim=1)  # (m, n_classes)
    transition_matrix = torch.full((n_classes, n_classes), 1/n_classes, dtype=torch.float32)

    for i in range(n_classes):
        mask = (labels == i)  
        if mask.sum() > 0:
            transition_matrix[i] = probs[mask].mean(dim=0)

    return transition_matrix


def to_super_class_matrix(sub_matrix, super_class_sizes):

    num_super_classes = len(super_class_sizes)
    super_class_matrix = torch.zeros((num_super_classes, num_super_classes))
    
    start_indices = [sum(super_class_sizes[:i]) for i in range(num_super_classes)]
    end_indices = [sum(super_class_sizes[:i+1]) for i in range(num_super_classes)]
    
    for i in range(num_super_classes):
        for j in range(num_super_classes):
            sub_block = sub_matrix[start_indices[i]:end_indices[i], start_indices[j]:end_indices[j]]
            super_class_matrix[i, j] = torch.sum(sub_block)
    
    row_sums = torch.sum(super_class_matrix, dim=1, keepdim=True)
    super_class_matrix = super_class_matrix / row_sums
    
    return super_class_matrix


def weighted_choice(probabilities, available_classes):
    total = torch.sum(probabilities)
    r = random.uniform(0, total.item())  
    upto = 0
    for i, prob in enumerate(probabilities):
        upto += prob
        if upto >= r:
            return available_classes[i]


def class_selection(T, C, n):

    selected_classes = []
    temp_C = copy.deepcopy(C)
    current_class = random.choice(temp_C)
    selected_classes.append(current_class)
    temp_C.remove(current_class)

    while temp_C:
        probabilities = T[current_class, :]
        available_probs = torch.tensor([probabilities[c] for c in temp_C])
        next_class = weighted_choice(available_probs + 1e-8, temp_C) 
        selected_classes.append(next_class)
        temp_C.remove(next_class)
        current_class = next_class

    group_split = [[] for i in range(n)]
    for i, cls in enumerate(selected_classes):
        group_split[i % n].append(cls)

    return group_split


def get_task_pool(task_d, task_c, domain_index_list, group_index_list, group_length_list, net=None, domain_specific_loader=None, device=None, mode='random'):
    
    task_pool = []
    domain_split = divide_list(shuffle_list(domain_index_list), task_d)

    
    for id in domain_split:
        if mode == 'random':
            group_split = divide_list(shuffle_list(group_index_list), task_c)

        elif mode == 'hard':
            net.eval()

            with torch.no_grad():
                input_sum = []
                label_sum = []
                for i in id: 
                    input, label = domain_specific_loader[i].next(all=True)

                    input = input.to(device)
                    label = label.to(device)
                    input_sum.append(input)
                    label_sum.append(label)

                input_sum = torch.cat(input_sum, dim=0)
                label_sum = torch.cat(label_sum, dim=0)

                logits = net.forward(input_sum)

                class_matrix = compute_transition_matrix(logits=logits, labels=label_sum)
                group_matrix = to_super_class_matrix(sub_matrix=class_matrix, super_class_sizes=group_length_list)
                group_split = class_selection(T=group_matrix, C=group_index_list, n=task_c)

        for ig in group_split:
            task_pool.append((id, ig))

    return shuffle_list(task_pool)