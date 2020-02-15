# -*- coding: utf-8 -*-

import numpy as np

class dtw:
    def __init__(self, x, y):
        try:
            assert len(x.shape) == 1
        except AssertionError:
            assert x.shape[1] == 1
            print('please reset the input shape, make sure it meet the requirement of input')
        self.x = x
        self.y = y
        self.loss, self.loss_mat = self.main()

    def distance(self, x, y, is_sum = False):
        sample_distance = lambda x, y: np.abs(x - y)
        
        if is_sum:
            return np.sum(sample_distance(x, y))
        else:
            return sample_distance(x, y)

    def main(self):
        self.org_mat = np.zeros((self.x.shape[0], self.y.shape[0]))
        t1,t2 = self.org_mat.copy(),self.org_mat.copy()
        for i in range(self.x.shape[0]):
            for j in range(self.y.shape[0]):
                t1[i,j] = self.x[i]
                
        for j in range(self.y.shape[0]):
            for i in range(self.x.shape[0]):
                t2[i,j] = self.y[j]
                
        self.org_mat = self.distance(t1, t2)
        cost_mat = self.org_mat.copy()
        
        for i in range(self.org_mat.shape[0]):
            for j in range(self.org_mat.shape[1]):
                if i > 0 and j > 0:
                    cost_mat[i,j] = cost_mat[i,j] + np.min([cost_mat[i - 1, j], cost_mat[i - 1,j - 1], cost_mat[i, j - 1]])
                elif i == 0 and j > 0:
                    cost_mat[i,j] = cost_mat[i,j] + np.min([cost_mat[i, j - 1]])
                    #print(np.min([cost_mat[i, j - 1]]))
                elif i > 0 and j == 0:
                    cost_mat[i,j] = cost_mat[i,j] + np.min([cost_mat[i - 1, j]])
        
        return cost_mat[i,j], cost_mat
    
if __name__ == "__main__":
    x = np.array([3,2,4,5,6,7,8,9,4,5,6,7,8,9,5,6,6])
    y = np.array([4,5,6,7,8,9,4,5,6,7,8,9,5,6,6])
    test = dtw(x,y)
    print(test.loss)
    print(test.loss_mat)
    print(test.org_mat)