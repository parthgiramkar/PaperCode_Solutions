import numpy as np

def softmax(x: np.ndarray) -> np.ndarray : 

    """
    Computes numerically stable softmax along the last axis.
    
    Args:
        x: Input logits (N, C)
        
    Returns:
        Probabilities (N, C)
    """

# This logic would-break(crash) for largerno. > about 750
 
    # summ = np.sum(np.exp(x),axis=1,keepdims=True)              # across_the sample
    # ans = np.exp(x) / summ

    max_value_across_sample = np.max(x,axis=1,keepdims=True)
    new_x = x - max_value_across_sample

    summ = np.sum(np.exp(new_x),axis=1,keepdims=True)              # across_the sample
    ans = np.exp(new_x) / summ
    return ans



if __name__ == "__main__" :
    

    x = [[749, 1.0, 0.1],       # Sample 0: class 0 has highest logit
    [0.5, 2.5, 0.3]   ]           # Sample 1: class 1 has highest logit

    print(f"{softmax(x)}")