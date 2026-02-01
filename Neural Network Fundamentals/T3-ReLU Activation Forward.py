import numpy as np



def relu_forward(x: np.ndarray) -> np.ndarray :
    
    """
    Computes ReLU(x) = max(0, x).
    """
    x = np.array(x) 

    ans = np.where( x<0 , 0 , x ,)               # return elements chosen from x or y depending on condition
    return ans


if __name__ == "__main__" :


    x =[-5,2,0]

    print(f"{relu_forward(x)}")
    