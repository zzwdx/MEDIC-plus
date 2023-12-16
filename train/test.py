import torch
from torch import nn
from util.log import log
from util.ROC import generate_OSCR
import torch.nn.functional as F
import copy


def calculate_acc(output, label):
    argmax = torch.argmax(output, axis=1)
    num_correct = (argmax == label).sum()
    return num_correct / len(output)


def generate_logits(net, loader, device="cpu"):
    net.eval()         

    output_sum = []
    b_output_sum = []
    label_sum = []  
    with torch.no_grad():  
        for input, label, *_ in loader:
            input = input.to(device)
            label = label.to(device)

            output = net(x=input)
            output = F.softmax(output, 1)
            b_output = net.b_forward(x=input)
            b_output = b_output.view(output.size(0), 2, -1)
            b_output = F.softmax(b_output, 1)

            output_sum.append(output)
            b_output_sum.append(b_output)
            label_sum.append(label)

    return torch.cat(output_sum, dim=0), torch.cat(b_output_sum, dim=0), torch.cat(label_sum)


def eval(net, loader, log_path, epoch=-1, device="cpu", mark="Val"):
    net.eval()
    criterion = nn.CrossEntropyLoss()

    num_correct = num_total = loss_val = 0

    for input, label, *_ in loader:
        input = input.to(device)
        label = label.to(device)

        output = net(x=input)
        loss = criterion(output, label)
        loss_val += loss.item() * len(input)
        argmax = torch.argmax(output, axis=1)
        num_correct += (argmax == label).sum()
        num_total += len(input)
    
    loss_avg = loss_val / num_total
    acc = num_correct / num_total

    log('Epoch: {} Loss: {:.4f} Acc: {:.4f} ({})'.format(epoch+1, loss_avg, acc, mark), log_path) 

    return acc


def eval_all(net, val_k, test_k, test_u, log_path, epoch=-1, device="cpu"):
    if val_k != None:
        output_v_sum, _, label_v_sum = generate_logits(net=net, loader=val_k, device=device)
        val_acc = calculate_acc(output_v_sum, label_v_sum)
        log('Epoch: {} Acc: {:.4f} ({})'.format(epoch+1, val_acc, "Val"), log_path) 
    else: 
        val_acc = 0

    output_k_sum, b_output_k_sum, label_k_sum = generate_logits(net=net, loader=test_k, device=device)  
    test_acc = calculate_acc(output_k_sum, label_k_sum)
    log('Epoch: {} Acc: {:.4f} ({})'.format(epoch+1, test_acc, "Test"), log_path) 

    if test_u != None:            
        output_u_sum, b_output_u_sum, *_ = generate_logits(net=net, loader=test_u, device=device)
        conf_k, argmax_k = torch.max(output_k_sum, axis=1)
        conf_u, _ = torch.max(output_u_sum, axis=1)

        oscr_c = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)
        log('Epoch: {} oscr_c: {:.4f} ({})'.format(epoch+1, oscr_c, "Test"), log_path) 


        _, argmax_k = torch.max(output_k_sum, axis=1)
        _, argmax_u = torch.max(output_u_sum, axis=1)

        argmax_k_vertical = argmax_k.view(-1, 1)
        conf_k = torch.gather(b_output_k_sum[:, 1, :], dim=1, index=argmax_k_vertical).view(-1)
        argmax_u_vertical = argmax_u.view(-1, 1)
        conf_u = torch.gather(b_output_u_sum[:, 1, :], dim=1, index=argmax_u_vertical).view(-1)

        oscr_b = generate_OSCR(argmax_k=argmax_k, conf_k=conf_k, label=label_k_sum, conf_u=conf_u)
        log('Epoch: {} oscr_b: {:.4f} ({})'.format(epoch+1, oscr_b, "Test"), log_path) 

    else:
        oscr_c = oscr_b = 0 

    return val_acc, test_acc, oscr_c, oscr_b


def update_recall(net, recall, log_path, model_val_path):

    if recall['va'] != 0:       
        if recall['va'] > recall['bva']:
            recall['bva'] = recall['va']
            recall['bvta'] = recall['ta']
            recall['bvt'] = [{
                "test_acc": "%.4f" % recall['ta'].item(),
                "oscr_c": "%.4f" % recall['oscrc'],
                "oscr_b": "%.4f" % recall['oscrb'],
            }]
            best_val_model = copy.deepcopy(net.state_dict())
            torch.save(best_val_model, model_val_path)
        elif recall['va'] == recall['bva']:
            recall['bvt'].append({
                "test_acc": "%.4f" % recall['ta'].item(),
                "oscr_c": "%.4f" % recall['oscrc'],
                "oscr_b": "%.4f" % recall['oscrb'],
            })
            if recall['ta'] > recall['bvta']:
                recall['bvta'] = recall['ta']
                best_val_model = copy.deepcopy(net.state_dict())
                torch.save(best_val_model, model_val_path)
        log("Current best val accuracy is {:.4f} (Test: {})".format(recall['bva'], recall['bvt']), log_path)
        
    if recall['ta'] > recall['bta']:
        recall['bta'] = recall['ta']   
        recall['btt'] = [{
            "oscr_c": "%.4f" % recall['oscrc'],
            "oscr_b": "%.4f" % recall['oscrb'],
        }]    
    log("Current best test accuracy is {:.4f} ({})".format(recall['bta'], recall['btt']), log_path)




