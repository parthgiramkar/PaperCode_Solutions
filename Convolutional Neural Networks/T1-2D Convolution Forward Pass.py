import numpy as np

def conv2d_forward(x, kernel, bias=None, stride=1, padding=0) :

    """
    x: np.ndarray of shape (N, C_in, H_in, W_in), dtype=np.float32
    kernel: np.ndarray of shape (C_out, C_in, K_h, K_w), dtype=np.float32
    bias: None or np.ndarray of shape (C_out,), dtype=np.float32
    stride: int or tuple of two ints (s_h, s_w)
    padding: int or tuple of two ints (p_h, p_w)
    returns: np.ndarray of shape (N, C_out, H_out, W_out), dtype=np.float32
    """

# getting the dimn's first
    batch_size , cin , inp_h , inp_w  = x.shape
    cout , cin , kernel_h , kernel_w = kernel.shape

# using this dimns ,calc. the dimns of op_matrix
    op_height = (inp_h + 2*padding - kernel_h ) // stride + 1
    op_width = (inp_w + 2*padding - kernel_w ) // stride + 1

# output matrix - 
    op_matrix = np.zeros( (batch_size , cout , op_height , op_width ) ) 


# as padding = 0 , i.e valid padding case , so need_to pad_the input

# main operation
    for n in range(batch_size) :

        current_image = x[n]

        for f in range(cout) :           # iterating over filter toget_each_feature map of an image

            current_featuremap = kernel[f]              
           # current_bias = b[f]          as given none

            for i in range(op_height) :

                for j in range(op_width) :
                    
                    ans = current_image[: , i*stride : i*stride+kernel_h , j*stride : j*stride+kernel_w]
                    res = ans * current_featuremap
        
                    op_matrix[n][f][i][j] = np.sum(res)

    return np.array(op_matrix,dtype=np.float32)




if __name__ == "__main__" :

# N=1 - means only 1 batch_size - only one image will_process_at_atime , Cin - 1=channel 
                    # means no rgb ,only_grayscale_img , H=3 - is_heightof_imag and W=3 -widthof_img

    x = np.array([[[[1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.]]]], dtype=np.float32)  # (1, 1, 3, 3)


# here filer , i.e Cout=1 , i.e no.of feature_map for each image , Cin = 1 , H=2 and W=2

    kernel = np.array([[[[1., 0.],
                        [0., 1.]]]], dtype=np.float32)      # (1, 1, 2, 2)

    print(f"{conv2d_forward(x, kernel, stride=1, padding=0)}")
    
 