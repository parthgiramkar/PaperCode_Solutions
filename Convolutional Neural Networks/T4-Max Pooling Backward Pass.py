import numpy as np


def max_pool_backward(dout, x, kernel_size, stride=None, padding=0) :

    """
    dout: np.ndarray of shape (N, C, H_out, W_out), dtype=np.float32
    x: np.ndarray of shape (N, C, H_in, W_in), dtype=np.float32
    kernel_size: int or tuple of two ints (K_h, K_w)
    stride: None, int, or tuple of two ints (s_h, s_w). If None, defaults to kernel_size.
    padding: int or tuple of two ints (p_h, p_w)
    returns: np.ndarray of shape (N, C, H_in, W_in), dtype=np.float32
    """
    stride_h , stride_w  = stride[0] , stride[1]
    
# dout - dl/dy : loss w.r.t to max_pool(X)
# dout = loss_func.backward_pass()

# now  , need to calcaute , dl/dx = dl/dy * dy/dx 
# gradient_X = Maxpool.backward_pass(dout)

    # as we need to calcaute loss of w.r.t x , the shape willbe same
    d_error_x = np.zeros( (x.shape) )
    
    batch , chann , h_in , w_in = d_error_x.shape   
    batch_size , chann , pool_h , pool_w = dout.shape          # the error map_w.rt. to max_pool(X)


    for n in range(batch_size) :
        for ch in range(chann ) :

            for i in range(pool_h) :

                for j in range(pool_w) :

                    x_patch = x[n , ch , i*stride_h:i*stride_h+pool_h , j*stride_w:j*stride_w+pool_w ]
                    maxi = np.max(x_patch)             # max_value that we need to calcaute error wrt it

                    dout_patch = dout[n,ch,i,j]             # single_value error

                    mask = ( maxi == x_patch )  # stores the max_value_as 1 and other as_0

                    summ = mask * dout_patch

                    d_error_x[n,ch,i*stride_h:i*stride_h+pool_h , j*stride_w:j*stride_w+pool_w] += summ
 
    return d_error_x

    raise NotImplementedError




if __name__ == "__main__" :


    x = np.array([[[[1., 2.],
                [3., 4.]]]], dtype=np.float32)  # (1, 1, 2, 2)

    dout = np.array([[[[1.]]]], dtype=np.float32)  # (1, 1, 1, 1) - gradient w.r.t. output

    dx = max_pool_backward(dout, x, kernel_size=2, stride=(2,2) , padding=0)
    # dx should have shape (1, 1, 2, 2)


    print(dx)




