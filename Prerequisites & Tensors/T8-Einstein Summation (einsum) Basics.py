# import numpy as np
# from typing import Dict, Union


# def einsum_ops(A: np.ndarray, B: np.ndarray) -> Dict[str, Union[np.ndarray, float]]  :


#     """
#     Computes basic ops using np.einsum.
    
#     Args:
#         A: (N, D)
#         B: (D, M)
        
#     Returns:
#         Dict with "transpose", "sum", "row_sum", "col_sum", "matmul"
#     """
    
#     A,B = np.array(A),np.array(B)
#     transp = np.einsum('ij->ji',A) 

#     sum = np.einsum('ij->',A)
    
#     row_sum = np.einsum('ij->i',A)

#     col_sum = np.einsum('ij->j',A)
#     print(transp,sum,row_sum,col_sum)

#     matmul = np.einsum('ij,jk->ik', A, B)
    
#     res : Dict[str,Union[np.ndarray,float]] = {}

#     res["transpose"] = transp
#     res["sum"] = sum
#     res["row_sum"] = row_sum
#     res["col_sum"] = col_sum
#     res["matmul"] = matmul


#     return res




# if __name__ == "__main__" :

#     A = [[1, 2],     # Shape (2, 2) - N=2, D=2
#      [3, 4]]
    
#     B = [[5, 6],     # Shape (2, 2) - D=2, M=2
#         [7, 8]]

#     A,B = np.array(A),np.array(B)

#     print(f"{einsum_ops(A,B)}")





 
# # Einstein summation notation(np.einsum()) follows this pattern: "input_indices -> output_indices"

# # Input indices: Label each dimension of input tensors (e.g., "ij" for a 2D matrix)
# # Output indices: Specify which indices appear in the output

# # | Operation | einsum String | Meaning |  - 
# # Transpose | "ij->ji" | Swap dimensions | 
# # | Sum all | "ij->" | Sum over all indices |
# # | Row sum | "ij->i" | Keep rows, sum columns | 
# # | Col sum | "ij->j" | Keep columns, sum rows | 
# # | Matrix multiply | "ik,kj->ij" | Contract shared dimension k |