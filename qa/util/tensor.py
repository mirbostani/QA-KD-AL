import torch

def tensor_from_var_2d_list(target, padding=0.0, max_len=None, requires_grad=True):
    """Convert a variable 2 level nested list to a tensor.
    e.g. target = [[1, 2, 3], [4, 5, 6, 7, 8]]
    """
    max_len_calc = max([len(batch) for batch in target])
    if max_len == None:
        max_len = max_len_calc
    
    if max_len_calc > max_len:
        print("Maximum length exceeded: {}>{}".format(max_len_calc, max_len))
        target = [batch[:max_len] for batch in target]

    padded = [batch + (max_len - len(batch)) * [padding] for batch in target]
    return torch.tensor(padded, requires_grad=requires_grad)