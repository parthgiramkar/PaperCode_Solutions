import numpy as np
from typing import Dict



def compute_norms(x: np.ndarray)  ->  Dict[str, np.ndarray]  :  

    """
    Computes L1 and L2 norms for a batch of vectors.
    
    Args:
        x: Input matrix of shape (N, D)
        
    Returns:
        Dictionary with keys "l1" and "l2", each containing an array of shape (N,)
    """
    
    abs_value = np.abs(x)

    L1_norm_Manhattan_distance = np.sum(abs_value,axis=1)         # as_need to calcaute across_the_row ,axis=1

    sqr_value = np.square(abs_value)
    L2_norm_euclidean_distance = np.sqrt(np.sum(sqr_value,axis=1))


    res : Dict[str,list] = {}

    res['l1'] = L1_norm_Manhattan_distance
    res['l2'] = L2_norm_euclidean_distance

    return res



if __name__ == "__main__" :

    x = [  [3, 4] ,         # First vector: (3, 4)
    [1, -1]  ]              # Second vector: (1, -1)

    print(f"{compute_norms(x)}")








# "l1": The L1 norm (Manhattan norm, taxicab norm) for each vector
# Formula :-  

# ‖x‖₁ = ∑_{i=1 to D} |x_i| ,           where D is dimn's ,i.e - across row


# "l2": The L2 norm (Euclidean norm) for each vector
# Formula:

# ‖x‖₂ = sqrt(∑_ sum_{i=1 to D} x_i^2 )  , across row

# A vector norm, sometimes represented with a double bar as ∥x∥, 
# is a function that assigns a non-negative length or size to a vector x in n-dimensional space

# l1 and 12 are used for measuring magnitudes or length_of vector
# while L∞ norm used for measuring the size of vector 