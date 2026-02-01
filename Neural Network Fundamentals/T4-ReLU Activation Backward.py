import numpy as np

def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray :

    """
    Computes dx for ReLU.
    """
    x = np.array(x)

    if len(dout) == len(x) :

        mask = np.where( x>0 , 1 , 0)
        dx = np.array(mask*dout,dtype=np.float32)
        return dx
    

    
if __name__ == "__main__" :

#    x = [0]

# # Backward pass
#    dout = [0.1]    # Upstream gradients
    x = [[-1, 2],
     [-3, 4]]

    dout = [[0.1, 0.2],
            [0.3,0.4]]

    print(f'{relu_backward(dout , x)}')