import numpy as np

def zero_pad(x, padding)  :


    """
    x: np.ndarray of shape (N, C, H, W), dtype=np.float32
    padding: int or tuple of two ints (p_h, p_w)
    returns: np.ndarray of shape (N, C, H+2*p_h, W+2*p_w), dtype=np.float32
    """
# as given padding only H and W , Hpads thepads top and bottom  while Wright and left corner

    # padded_x = np.pad ( x , pad_width = ( (0,0) , (0,0) ,  (padding,padding) , (padding,padding)  ) , mode='constant' , constant_values=0 )
    # return padded_x              # new matrix shape , H+2*p_h, W+2*p_w

    # raise NotImplementedError


# Manually


def reflect_pad(x, padding)  :


    """
    x: np.ndarray of shape (N, C, H, W), dtype=np.float32
    padding: int or tuple of two ints (p_h, p_w)
    returns: np.ndarray of shape (N, C, H+2*p_h, W+2*p_w), dtype=np.float32
    """

# 'edge' - Pads with the edge values of array.
    # reflect_pad = np.pad( x , pad_width = ( (0,0) , (0,0) ,  (padding,padding) , (padding,padding)  ) , mode='edge')
    # return reflect_pad

    # raise NotImplementedError


# Manually





if __name__ == "__main__" :


    x = np.array([
[       [
        [1., 2.],
        [3., 4.]
                ]
]
                
    ], dtype=np.float32)  # (1, 1, 2, 2)


    # Zero padding
    out_zero = zero_pad(x, padding=1)
    print("Zero padding",out_zero)

    # Reflection padding
    out_reflect = reflect_pad(x, padding=1)
    print("reflect padding",out_reflect)