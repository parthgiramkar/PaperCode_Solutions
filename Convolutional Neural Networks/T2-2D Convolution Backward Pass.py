import numpy as np

def conv2d_backward(dout, x, kernel, bias=None, stride=1, padding=0) :


    """
    dout: np.ndarray of shape (N, C_out, H_out, W_out), dtype=np.float32
    x: np.ndarray of shape (N, C_in, H_in, W_in), dtype=np.float32
    kernel: np.ndarray of shape (C_out, C_in, K_h, K_w), dtype=np.float32
    bias: None or np.ndarray of shape (C_out,), dtype=np.float32
    stride: int or tuple of two ints (s_h, s_w)
    padding: int or tuple of two ints (p_h, p_w)
    returns: tuple (dx, dkernel, dbias) where:
        dx: np.ndarray of shape (N, C_in, H_in, W_in), dtype=np.float32
        dkernel: np.ndarray of shape (C_out, C_in, K_h, K_w), dtype=np.float32
        dbias: np.ndarray of shape (C_out,) or None, dtype=np.float32
    """

    raise NotImplementedError



if __name__ == "__main__" :

    x = np.array([[[[1., 2.],
                [3., 4.]]]], dtype=np.float32)  # (1, 1, 2, 2)

    kernel = np.array([[[[1., 1.],
                        [1., 1.]]]], dtype=np.float32)  # (1, 1, 2, 2)

    dout = np.array([[[[1., 1.],
                    [1., 1.]]]], dtype=np.float32)  # (1, 1, 2, 2) - gradient w.r.t. output

    dx, dkernel, dbias = conv2d_backward(dout, x, kernel, stride=1, padding=0)
    # dx should have shape (1, 1, 2, 2)
    # dkernel should have shape (1, 1, 2, 2)