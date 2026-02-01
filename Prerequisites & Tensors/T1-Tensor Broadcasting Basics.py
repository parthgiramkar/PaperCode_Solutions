import numpy as np

def broadcast_ops(X: np.ndarray, b: np.ndarray, w: np.ndarray) ->  np.ndarray  :


    """
    Computes (X + b) * w using broadcasting.
    
    Args:
        X: Input matrix of shape (N, D)
        b: Bias vector of shape (D,)
        w: Weight vector of shape (N,)
        
    Returns:
        Resulting matrix of shape (N, D)
    """


    X , b , w = np.array(X) , np.array(b) , np.array(w)
    
    print(X.shape , b.shape , w.shape)                    # as_here the rule_match for adding matrix_elemntwise,

# so no_need of broadcasting vector , simple - (N,D) + (D,)
    sum = X + b
    print(sum)                      # shape - (N,D)

    # print(sum * w)                   # here ,the dimensions not matching- (N,D) * (N,) 

# To perform element-wise operation , need to match - (N,D) this with (N,) , so need to be reshaped/broadcasted

    w = np.reshape( w,(-1,1))         # we need coln vector to_go element_wise operation
    print(w,w.shape)
    return sum * w                # now the shapes are matching - (N,D) with (N,1)




if __name__ == "__main__" :


# matrix 
    X = [ [1, 2],
        [3, 4],
        [5, 6] ]
# vector 
    b   = [10, 100]
    w = [0.5, 2.0, -1.0]

    print(f"After performing Brodcasting operation - {broadcast_ops(X,b,w)}")






# Broadcasting is the mechanism that allows element-wise
# operations on arrays of different shapes by "stretching" the smaller array to match the larger one.










