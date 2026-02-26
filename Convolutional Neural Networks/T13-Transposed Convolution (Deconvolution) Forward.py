import numpy as np

def transposed_conv2d_forward(x, kernel, bias=None, stride=1, padding=0, output_padding=0) :

    """
    Transposed convolution (deconvolution): upsampling convolution.
    
    x: np.ndarray of shape (N, C_in, H_in, W_in), dtype=np.float32
    kernel: np.ndarray of shape (C_in, C_out, K_h, K_w), dtype=np.float32
    bias: None or np.ndarray of shape (C_out,), dtype=np.float32
    stride: int or tuple
    padding: int or tuple
    output_padding: int or tuple
    returns: np.ndarray of shape (N, C_out, H_out, W_out), dtype=np.float32
    """

    if stride > 1 :

    
    padded_z = np.pad





    raise NotImplementedError






if __name__ == "__main__" :



    x = np.array([[[[1., 2.],
                [3., 4.]]]], dtype=np.float32)  # (1, 1, 2, 2)

    kernel = np.ones((1, 1, 3, 3), dtype=np.float32) * 0.1
    bias = np.zeros(1, dtype=np.float32)

    out = transposed_conv2d_forward(x, kernel, bias, stride=2, padding=1)
    # Output will be larger than input (upsampled)
    # Shape depends on stride, padding, and kernel size
    print(out)