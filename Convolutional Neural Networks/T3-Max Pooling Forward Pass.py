import numpy as np

def max_pool_forward (x , kernel_size, stride = None , padding =0 ) :


# getting the dimn's first
        batch_size , cin , inp_h , inp_w  = x.shape
        kh , kw = kernel_size[0] , kernel_size[1]           # size of the window
        sh ,sw = stride[0] , stride[1]
        ph , pw = padding[0] , padding[1]

# using this dimns ,calc. dimns of op_matrix
        op_height = (inp_h + 2*ph - kh ) // sh + 1
        op_width = (inp_w + 2*pw - kw ) // sw + 1

# output matrix - 
        polled_matrix = np.zeros( (batch_size , cin , op_height , op_width ) ) 

# padding the input
        padded_ip = np.pad(x , pad_width = ( (0,0) ,(0,0) , (padding) , (padding) ) ,mode='constant',constant_values=0)



# main operation , now performing on the padded_ip

        for n in range(batch_size) :

            current_image = padded_ip[n]

            for f in range(cin) :           # iterating over filter toget_each_feature map of an image            
            # current_bias = b[f]          as given none

                for i in range(op_height) :

                    for j in range(op_width) :
                        
                        ans = current_image[f , i*sh : i*sh+kh , j*sw : j*sw+kw]            
                        polled_matrix[n][f][i][j] = np.max(ans)


        return np.array(polled_matrix,dtype=np.float32)





if __name__ == "__main__" :
     
    x = np.array([[[[1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.]]]], dtype=np.float32)  # (1, 1, 4, 4)

    out = max_pool_forward(x, kernel_size=(2,2), stride=(2,2), padding=(0,0))
    print(out)





