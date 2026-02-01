import numpy as np
from typing import Dict



def linear_backward(dout: np.ndarray, x: np.ndarray, w: np.ndarray, b: np.ndarray) -> Dict[str, np.ndarray]  :


    """
    Computes dx, dw, db for y = x @ w + b.
    
    Args:
        dout: Upstream gradient (N, Dout)
        x: Input (N, Din)
        w: Weights (Din, Dout)
        b: Bias (Dout,)
        
    Returns:
        Dict with "dx", "dw", "db"
    """
    x,b,w,dout = np.array(x),np.array(b) , np.array(w) ,np.array(dout)
    print(x.shape,b.shape,w.shape,dout.shape)


# Gradient with respect to input X , i.e how loss L changes w.rt. each_element_of output Y

    # Y = np.matmul(x,w) + b 
    # print(Y , Y.shape)

    dx = dout @ w.T                         # transposed to_match_the dimensions
    print(dx ,dx.shape)


# Gradient with respect to weights W ,i.e how loss L changes w.rt. each_weight
    dw = x.transpose() @ dout                      # transposed to_match_the dimensions
    print(dw ,dw.shape)



# Gradient with respect to bias B , i.e how loss L changes w.rt. each_bias
    db = np.sum(dout , axis=0)                  # as bias is added per feature_col/ per op_neuron                  
    print(db,db.shape)


    res : Dict[str , np.ndarray] = {}

    res['dx'] = dx
    res['dw'] = dw
    res['db'] = db


    return res



if __name__ == "__main__" :

            
    x = [[1, 2],
    [3, 4],
    [5, 6] ]          # (3,2)

    w = [[1, 0, 2, 1],                          # 4_o/p neurons for each feature
        [0, 1, 3, 2]]    # (2,4)

    b = [1, 1, 1, 1]      # (4,)                # for each neuron , each bias is added

    dout = [[0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
            [0.9, 1.0, 1.1, 1.2]]   # (3,4)


    print(f"{linear_backward(dout,x,w,b)}")














