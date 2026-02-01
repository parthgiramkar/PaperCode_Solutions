import numpy as np
from typing import Dict


def grad_matmul(grad_C: np.ndarray, A: np.ndarray, B: np.ndarray) -> Dict[str, np.ndarray] :


    """
    Computes gradients for C = AB.
    
    Args:
        grad_C: Upstream gradient dL/dC (M, N)
        A: Input A (M, K)
        B: Input B (K, N)
        
    Returns:
        Dict with "grad_A" and "grad_B"
    """
    A,B,grad_C = np.array(A) , np.array(B),np.array(grad_C)

    gradient_A = grad_C @ B.T
    print(gradient_A)

    gradient_B = A.transpose() @ grad_C
    print(gradient_B)

    res : Dict[str,np.ndarray] = {}

    res['grad_A'] = gradient_A

    res['grad_B'] = gradient_B

    return res



if __name__ == "__main__" :

        
    # Forward pass
    A = [[1, 2],     # Shape (2, 2) - M=2, K=2
    [3, 4] ]
    
    B = [[5, 6],     # Shape (2, 2) - K=2, N=2
        [7, 8]]
    
    # C = A @ B = [[19, 22],   # Shape (2, 2) - M=2, N=2
    #             [43, 50]]
    

    # Backward pass
    grad_C = [[0.1, 0.2],    # Upstream gradient (M×N)
            [0.3, 0.4]]


    print(f"{grad_matmul(grad_C , A , B)}")









































