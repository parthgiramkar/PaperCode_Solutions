import numpy as np


def cnn_block_forward(x, kernel, bias, conv_stride=1, conv_padding=0, pool_size=2, pool_stride=None, pool_padding=0):

    """
    x: np.ndarray of shape (N, C_in, H_in, W_in), dtype=np.float32
    kernel: np.ndarray of shape (C_out, C_in, K_h, K_w), dtype=np.float32
    bias: None or np.ndarray of shape (C_out,), dtype=np.float32
    conv_stride: int or tuple of two ints
    conv_padding: int or tuple of two ints
    pool_size: int or tuple of two ints
    pool_stride: None, int, or tuple of two ints
    pool_padding: int or tuple of two ints
    returns: np.ndarray, dtype=np.float32
    """

# Setting the dimensions

    batch_size , channels , h_in , w_in = x.shape
    filters , channels , k_h , k_w = kernel.shape     # channels are same_for weight,bais and inp
    channels = bias[0]
    pad_h , pad_w = conv_padding[0] , conv_padding[1]
    stride_h , stride_w  = conv_stride[0] , conv_stride[1]
    pool_h , pool_w = pool_size[0] , pool_size[1]
    pool_stride_h , pool_stride_w  = pool_stride[0] , pool_stride[1]

    pool_pad_h ,  pool_pad_w = pool_padding[0], pool_padding[1]

# getting op_matrix - Z = inp*W + bias

    z_h = ( h_in + 2*pad_h - k_h ) // stride_h + 1
    z_w =  ( w_in + 2*pad_h - k_w ) // stride_w + 1

    z = np.zeros( (batch_size , filters , z_h,z_w) ) 

    padded_x = np.pad(x  , pad_width = ( (0,0) , (0,0) , (pad_h,pad_h) , (pad_w,pad_w) ) ,
                    mode='constant' , constant_values=0 ) 
    
# operation - Convulation = X1 - conv2d(X,K,b)
    for n in range(batch_size) :

        for f in range(filters) :
            current_filter = kernel[f]
            current_bias = bias[f]
            for i in range(z_h) :
                for j in range(z_w) :

                    inp_patch = padded_x[n,:,i*stride_h : i*stride_h+k_h , j*stride_w : j*stride_w+k_w]
                    slide = np.sum(inp_patch * current_filter) + current_bias
                    z[n][f][i][j] = slide


# operation - Relu activation = X2 = Relu(X1)
    relu = np.zeros( (z.shape) ) 
    relu = np.where(z>0,z,0)
    # for n in range(batch_size) :

    #     for f in range(filters) :
           
    #         for i in range(z_h) :
    #             for j in range(z_w) :

    #                 ans = z[n,f,i,j]
    #                 if ans > 0 :
    #                     relu[n][f][i][j] = ans
    #                 else :
    #                     relu[n][f][i][j] = 0


# operation - Maxpool(X2)
#  now shrinked down the feature_map using maxpooling , to pooling_window of size 2X2
    padded_relu  = np.pad( relu , pad_width=( (0,0) , (0,0) , (pool_pad_h,pool_pad_h) , (pool_pad_w ,pool_pad_w )) , mode='constant' )

# sliding pool_h and _w

    h_out = (z_h  + 2*pool_pad_h - pool_h ) // pool_stride_h + 1
    w_out = (z_w + 2*pool_pad_w - pool_w ) // pool_stride_w + 1

    pooled_op = np.zeros( (batch_size , filters ,h_out , w_out ) )

    for n in range(batch_size) :

        for f in range(filters) :
            
            for i in range(h_out) :
                for j in range(w_out) :

                    patch = padded_relu[n,f,i*pool_stride_h: i*pool_stride_h+pool_h , j*pool_stride_w:j*pool_stride_w+pool_w  ]
                    maxi = np.max(patch)
                    pooled_op[n][f][i][j] = maxi

    return pooled_op


    raise NotImplementedError






if __name__ == "__main__" :



    x = np.array([[[[1., 2., 3., 4.],
                    [5., 6., 7., 8.],
                    [9., 10., 11., 12.],
                    [13., 14., 15., 16.]]]], dtype=np.float32)  # (1, 1, 4, 4)

    kernel = np.array([[[[1., 1.],
                        [1., 1.]]]], dtype=np.float32)  # (1, 1, 2, 2)

    out = cnn_block_forward(x, kernel,bias=(1,) ,conv_stride=(1,1), conv_padding=(0,0), pool_size=(2,2), pool_stride=(2,2) ,  pool_padding = (0,0) )
    # Applies: conv → relu → max_pool
    print(out)