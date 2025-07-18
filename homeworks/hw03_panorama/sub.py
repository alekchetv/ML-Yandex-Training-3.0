import numpy as np
# do not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
# __________end of block__________


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    des1_sum = np.sum(des1**2, axis=1, keepdims=True)
    des2_sum = np.sum(des2**2, axis=1, keepdims=True).T
    mul = np.matmul(des1, des2.T)
    ln = np.sqrt(des1_sum - 2 * mul + des2_sum)
    indexes_des1_to_des2 = np.argmin(ln, axis=1)
    indexes_des2_to_des1 = np.argmin(ln, axis=0)
    
    matches = []
    for i in range(len(des1)):
        j = indexes_des1_to_des2[i]
        if indexes_des2_to_des1[j] == i:
            matches.append(DummyMatch(i, j, float(ln[i, j])))
    
    return sorted(matches, key = lambda x:x.distance)

