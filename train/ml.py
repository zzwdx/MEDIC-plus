import torch
import inspect


def load_fast_weights(net, fast_parameters):
    for weight, fast in zip(net.parameters(), fast_parameters or [None] * len(list(net.parameters()))):
        weight.fast = fast


def compute_gradient(net, fast_parameters, input, label, criterion, ovaloss, weight):
    out_c, out_b = net.c_forward(x=input)
    out_b = out_b.view(out_b.size(0), 2, -1)

    loss = criterion(out_c, label) + ovaloss(out_b, label)
    loss *= weight

    grad = torch.autograd.grad(loss, fast_parameters, create_graph=False, allow_unused=True)
    grad = [g.detach() if g is not None else g for g in grad]

    return grad


def update_fast_weights_reptile(net, grad, meta_lr):
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

    return fast_parameters


def update_fast_weights_sam(net, grad, meta_lr):
    fast_parameters = []
    for k, weight in enumerate(net.parameters()):
        if grad[k] is not None:
            if weight.fast is None:
                weight.fast = weight + meta_lr * grad[k]
            else:
                weight.fast = weight.fast + meta_lr * grad[k]

        if weight.fast is None:
            fast_parameters.append(weight)
        else:
            fast_parameters.append(weight.fast)

    return fast_parameters


update_methods = {
    "reptile": update_fast_weights_reptile,
    "sam": update_fast_weights_sam,
}


def update_fast_weights(method_name, **kwargs):
    if method_name not in update_methods:
        raise ValueError(f"Unknown method: {method_name}")
    
    method = update_methods[method_name]
    
    sig = inspect.signature(method)
    filtered_kwargs = {
        k: v for k, v in kwargs.items() if k in sig.parameters
    }
    
    return method(**filtered_kwargs)


def accumulate_meta_grads_reptile(net, meta_lr):
    for weight in net.parameters():
        if weight.fast is not None:
            new_grad = (weight - weight.fast) / meta_lr
            if weight.grad is None:
                weight.grad = new_grad
            else:
                weight.grad += new_grad


def accumulate_meta_grads_arith(net, grad, meta_lr, eta):
    scale = eta / meta_lr
    for weight, g in zip(net.parameters(), grad):
        if g is not None:
            if weight.grad is None:
                weight.grad = g * scale
            else:
                weight.grad += g * scale


accumulate_methods = {
    "reptile": accumulate_meta_grads_reptile,
    "arith": accumulate_meta_grads_arith,
}


def accumulate_meta_grads(method_name, **kwargs):
    if method_name not in accumulate_methods:
        raise ValueError(f"Unknown method: {method_name}")
    
    method = accumulate_methods[method_name]
    
    sig = inspect.signature(method)
    filtered_kwargs = {
        k: v for k, v in kwargs.items() if k in sig.parameters
    }
    
    return method(**filtered_kwargs)