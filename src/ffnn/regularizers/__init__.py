from .Regularizer import Regularizer, L1, L2

REGULARIZER_REGISTRY = {
    "l1": L1,
    "l2": L2,
}

def get_regularizer(name, **kwargs):
    """
    **kwargs: Parameters to pass to the regularizer
    """
    if name is None:
        return None
    
    if name not in REGULARIZER_REGISTRY:
        raise ValueError(
            f"Unknown regularizer: {name}. "
            f"Available: {list(REGULARIZER_REGISTRY.keys())}"
        )
    return REGULARIZER_REGISTRY[name](**kwargs)


__all__ = ["Regularizer", "L1", "L2", "get_regularizer"]