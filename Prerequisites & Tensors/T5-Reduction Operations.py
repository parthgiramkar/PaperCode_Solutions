
import numpy as np
from typing import Dict, Union


def tensor_reductions(x: np.ndarray, axis: int) -> Dict[str, Union[np.ndarray, float]] :

    """
    Computes sum, mean, max, argmax along axis.
    """
    x = np.array(x)

    summ = np.sum( x,axis=axis )
    mean = np.mean( x,axis=axis )
    max = np.max( x , axis=axis  )
    argmax = np.argmax(x , axis= axis )                 # index of the maximum values along an axis 
    print(summ.shape , mean.shape)
    res : Dict[str, Union[np.ndarray, float]]  = {}

    res["sum"] = summ
    res["mean"] = mean
    res["argmax"] = argmax
    res["max"] = max 

    return res


if __name__ == "__main__" :



    x = [[1, 2, 3], 
    [4, 5, 6]  ]             # Shape (2, 3)

    axis = 1  # Reduce along columns (second dimension)

    print(f"{tensor_reductions(x,axis)}")