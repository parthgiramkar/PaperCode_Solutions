# import numpy as np
# from typing import Dict

# def sigmoid_ops(x: np.ndarray, dout: np.ndarray) -> Dict[str, np.ndarray] :

#     """
#     Computes sigmoid forward and backward.
#     """
# #     x = np.array(x)
# # # calcaute_sigmoid_first

# #     out = 1 / ( 1 + np.exp(-x))
# #     print(out)

# #     sigma_prime = out * ( 1 - out )
# #     print(sigma_prime)

# #     dx = dout * sigma_prime
# #     print(dx)
# #     res :  Dict[str, np.ndarray] = {}
    
# #     res['out'] = out
# #     res['dx'] = dx

# #     return res


# if __name__ == "__main__" :

#     x = [-2.0, -1.0, 0.0, 1.0, 2.0]

#     # Backward pass
#     dout = [0.1, 0.2, 0.3, 0.4, 0.5]    

#     print(f"{sigmoid_ops(x,dout)}")