import torch


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def norm(x):
    out = (x -0.5) *2
    return out.clamp(-1, 1)

def move_to_gpu(t, opt):
    if opt.cuda:
        t = t.to(opt.device)
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t