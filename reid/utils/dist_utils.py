import torch


@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensors_gather = [
            torch.ones_like(tensor)
            for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        for i in range(len(tensors_gather)):
            print(tensors_gather[i].size())
        output = torch.cat(tensors_gather, dim=0)

        return output
    else:
        return tensor
