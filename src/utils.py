from torch.utils.data.dataloader import default_collate

def get_collate(name):
    if name == "identity":
        return lambda x: x
    else:
        return default_collate  

def t2im(t):
    if len(t.shape) == 3:
        t = t[None, ...]
    elif len(t.shape) == 2:
        t = t[None, None, ...]
    if t.shape[1] == 1:
        t = t.repeat(1, 3, 1, 1)
    b, c, h, w = t.shape
    t = t.data.cpu().numpy()
    t = t.clip(-1, 1)
    t = (t + 1) / 2
    t = t * 255
    t = t.transpose((2, 0, 3, 1)).reshape((h, w*b, c))
    return np.ascontiguousarray(t).astype('uint8')