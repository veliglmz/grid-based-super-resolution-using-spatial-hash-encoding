import cv2
import numpy as np
from scipy.optimize import least_squares


sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

def func(mat, x0, x1):
    mat = np.reshape(mat, (2, 3))
    x_prime = mat @ x0.T
    return np.sum((x1 - x_prime.T)**2, axis=1)

def reject_outliers(data, m=6.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]

def eliminate_matches(matches):
    # get the matches includes at least 6 matches
    slctd = []
    rt = 0.2
    for _ in range(6):
        good = []
        for m,n in matches:
            if m.distance < (rt * n.distance):
                good.append(m)
        if len(good) < 10:
            rt += 0.05
        else:
            slctd = [elem for elem in good]
            break
    if len(slctd) < 6:
        raise Exception(f"Number of matched keypoints is {len(slctd)}, less than 6!")
    return slctd

def cal_affine_matrix(img1, img2):

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    kps1, descs1 = sift.detectAndCompute(img1, None)
    descs1 /= (descs1.sum(axis=1, keepdims=True) + 1e-7)
    descs1 = np.sqrt(descs1)

    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kps2, descs2 = sift.detectAndCompute(img2, None)
    descs2 /= (descs2.sum(axis=1, keepdims=True) + 1e-7)
    descs2 = np.sqrt(descs2)

    matches = bf.knnMatch(descs1, descs2, k=2)
    matches = eliminate_matches(matches)
    matches = sorted(matches, key=lambda x : x.distance)

    m_kps1 = []
    m_kps2 = []
    for m in matches:
        m1 = m.queryIdx
        m2 = m.trainIdx
        m_kps1.append(kps1[m1].pt)
        m_kps2.append(kps2[m2].pt)
    
    m_kps1 = np.asarray(m_kps1, dtype=np.float32)
    m_kps2 = np.asarray(m_kps2, dtype=np.float32)

    # gets 3 keypoints having the minimum distance
    warp_mat = cv2.getAffineTransform(m_kps2[:3], m_kps1[:3])
    
    warp_mat = warp_mat.flatten()
    n, _ = m_kps1.shape
    ones = np.ones((n,1))
    m_kps2 = np.hstack((m_kps2, ones))
    
    warp_mat = least_squares(func, warp_mat, args=(m_kps2, m_kps1), method="lm")
    warp_mat = np.reshape(warp_mat.x, (2, 3))

    #warp_dst = cv2.warpAffine(img2, warp_mat, (img2.shape[:2]), flags=cv2.INTER_CUBIC)

    return warp_mat
