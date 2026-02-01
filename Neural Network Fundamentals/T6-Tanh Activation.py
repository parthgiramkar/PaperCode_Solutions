# import numpy as np
# from typing import Dict


# def tanh_ops(x: np.ndarray, dout: np.ndarray) -> Dict[str, np.ndarray] : 

#     """
#     Computes tanh forward and backward.
#     """
#     out = np.tanh(x)
#     print(out)

#     tanh_prime = 1 - (out)**2
#     print(tanh_prime)  

#     dx = dout*tanh_prime
#     res :  Dict[str, np.ndarray] = {}
    
#     res['out'] = out
#     res['dx'] = dx

#     return res



# if __name__ == "__main__" :

#     # Forward pass
#     x = [-2.0, -1.0, 0.0, 1.0, 2.0]

#     # Backward pass
#     dout = [0.1, 0.2, 0.3, 0.4, 0.5]

#     print(f"{tanh_ops(x,dout)}")



