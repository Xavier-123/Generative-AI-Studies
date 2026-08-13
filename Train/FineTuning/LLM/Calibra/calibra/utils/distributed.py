"""Optional distributed helpers that are safe in single-process runs."""


def is_distributed() -> bool:
    try:
        import torch.distributed as dist
        return dist.is_available() and dist.is_initialized()
    except ImportError:
        return False


def barrier() -> None:
    if is_distributed():
        import torch.distributed as dist
        dist.barrier()
