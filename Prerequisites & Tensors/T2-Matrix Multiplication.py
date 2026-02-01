import numpy as np


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:

    """
    Computes matrix product C = AB using 3 nested loops.
    """

    A , B = np.array(A) , np.array(B)
    m , k = A.shape
    j , n = B.shape

    ans = np.zeros( (m,n) )         # o/p matrix

    if k == j :

        for x in range(m) :

            for y in range(n) :

                for z in range(k) :              # size of depth(downwards going) , i.e size of K or j

                    ans[x][y] += A[x][z]*B[z][y]
        
        return ans 

    else :

        return "Dimensions not compatible"


def matmul_vectorized(A: np.ndarray, B: np.ndarray) -> np.ndarray:



    """
    Computes matrix product C = AB using vectorized operations.
    """

    A , B = np.array(A) , np.array(B)
    m , k = A.shape   
    j , n = B.shape 

    if k == j :

        ans = A @ B                             # np.matmul() same as using @ operator
        #ans = np.dot(A,B)                          # if matrix 1or2-D ,np.dot() yields same result,but not for > 2 dimens 
        return ans
    
    else :

        return "Dimensions not compatible"
    


if __name__ == "__main__" :


    A = [
        [1, 2, 3],
        [4, 5, 6]
    ]    
    B =  [
        [7, 8],
        [9, 10],
        [11, 12]
    ] 
    

    print(f"With nested loops - {matmul_naive(A,B)}")
    print(f"WIthout(vectorised) - {matmul_vectorized(A,B)}")


