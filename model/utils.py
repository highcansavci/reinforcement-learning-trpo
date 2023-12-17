import torch
import torch.nn as nn

def init_weights(net, init='norm', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net


def init_model(model):
    model = init_weights(model, init="xavier")
    return model


def init_model_(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    grad_ = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    grad_ = torch.cat([t.view(-1) for t in grad_])
    return grad_


def conjugate_gradient(Ax, b, tol=1e-10, max_iterations=10):
    x = torch.zeros_like(b)
    r, p = b.clone(), b.clone()
    for i in range(max_iterations):
        a_vector_product = Ax(p)
        alpha = torch.dot(r, r) / torch.dot(p, a_vector_product)
        x += alpha * p
        r -= alpha * a_vector_product
        if torch.norm(p) < tol:
            break
        beta = torch.dot(r, r) / torch.dot(r - alpha * a_vector_product, r)
        p = r + beta * r
    return x


def apply_update(grad_flattened, neural_net):
    n = 0
    for param in neural_net.parameters():
        numel_ = param.numel()
        grad_ = grad_flattened[n:n+numel_].view(param.shape)
        param.data += grad_
        n += numel_




