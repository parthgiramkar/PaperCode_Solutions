import numpy as np
from typing import Dict



def elementwise_ops(a: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]:

    """
    Computes element-wise add, mul, and safe div.
    
    Args:
        a: First tensor
        b: Second tensor (same shape)
        
    Returns:
        Dictionary with keys "add", "mul", "div"
    """
    epsilon = 1e-8


# operations performed elementwise

    a,b = np.array(a) , np.array(b)

    res : Dict[str] = {}

    res["add"] = a + b
    res["mul"] = a*b
    res["div"] = a / (b +epsilon)

    return res


if __name__ == "__main__" :

    a = [1.0, 2.0]  # Shape (2,)
    b = [0.0, 2.0]  # Shape (2,)

    print(f"{elementwise_ops(a,b)}")




# Adding ϵ ensures numerical stability without significantly affecting 
# the result when the denominator is non-zero.


