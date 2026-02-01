import numpy as np

def linear_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray :

    """
    Computes y = x @ w + b
    
    Args:
        x: (N, Din)
        w: (Din, Dout)
        b: (Dout,)
        
    Returns:
        y: (N, Dout)
    """
    x,w,b = np.array(x),np.array(w),np.array(b)
    print(x.shape,w.shape,b.shape)

    # we_need to implement this - Y=XW+b

    if x.shape[1] == w.shape[0] :

        Y = x @ w 
        #print( Y)                                          # shape - (n,d_out)
        Y += b                          # numpy automatically brodcasted (d_out) to (1,d_out)
        return Y

    else :
        return ValueError("Dimensions are not compatible")



if __name__ == "__main__" :


    # Input: 1 sample with 2 features
    x = [[1,]]           # Shape (1, 2) - N=1, D_in=2

    # Weights: transform 2 inputs to 2 outputs
    w = [[1, 0],           # Shape (2, 2) - D_in=2, D_out=2
        [0, 1]]           # Identity-like transformation

    # Bias: shift each output
    b = [1, 1]  

    print(f"{linear_forward(x,w,b)}")