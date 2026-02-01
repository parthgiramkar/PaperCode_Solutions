import numpy as np
from typing import Tuple


def grad_sum(grad_y : float , x_shape : Tuple[int] ) ->  np.ndarray  :

    """
    Computes gradient of x given gradient of sum(x).
    
    Args:
        grad_y: Scalar gradient dL/dy
        x_shape: Shape of the input tensor x
        
    Returns:
        Gradient dL/dx of shape x_shape
    """

# diff y w.r.t to x i.e-(xi) - d(x1+x2+x3) / dxi - so,its 1 for each_of xi(variables)

    dy_wrt_xi =  np.ones( (x_shape))
    print(dy_wrt_xi)


# diff loss w.r.t to xi - dl/dxi =  dl/dy * dy/dxi , where dl/dy given
    gradient = grad_y * dy_wrt_xi
    print(gradient) 

    return gradient



if __name__ == "__main__" :

    x = [2.0, 3.0, 5.0]  # Shape (3,)
    #y = sum(x) = 10.0    # Scalar

    x = np.array(x)
    print(x.shape)

    x_shape = (3,)
    # Backward pass
    grad_y = 0.5  # Upstream gradient: ∂L/∂y = 0.5

    print(f"{grad_sum(grad_y , x_shape)}")




