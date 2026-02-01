import numpy as np

def reshape_and_transpose(x: np.ndarray, B: int, C: int, H: int, W: int) ->   np.ndarray  :

    """
    Reshapes flat x to (B, C, H, W) then transposes to (B, H, W, C).
    """
    
    np.reshape()






if __name__ == "__main__"  :


    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

    # Parameters
    B = 1  # 1 image in batch
    C = 2  # 2 channels
    H = 3  # 3 rows (height)
    W = 4  # 4 columns (width)


    print(f"{reshape_and_transpose(x,B,C,H,W)}")