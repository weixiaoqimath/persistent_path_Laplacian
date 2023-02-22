# As of Python 3.7, "Dict keeps insertion order" is the ruling.
# So please use Python 3.7 or later version.
import numpy as np
import gudhi
import scipy.spatial.distance
import os
from operator import itemgetter
import copy

def bdryOA(mat, num_rows, num_A_t0):
    """
    This function can do a column reduction with respect to certain row indices.
    parameter
    _________
    mat: a matrix. will be the 'boundary' matrix whose domain is allowed p+1 paths and 
    codomain is regular p paths (indeed \partial(A_{p+1}), a subset of regular p paths).
    num_rows: in the algorithm it is the dimension of allowed p paths
    num_A_t0: the dimension of allowed p+1 paths with weight <= t0
    Output
    ------
    the output mat[-m:, O_indices] is the tranformation matrix from \partial invariant p+1 paths to allowed p+1 paths.
    Column reduction.
    """
    mat = mat.astype('float64')
    n, m = mat.shape 
    # in this case, \partial(Omega_{p+1}) is a subset of A_p and Omega_{p+1} = A_{p+1}
    if num_rows == n:
        return mat, np.identity(m), num_A_t0, m
    I = np.identity(m)
    mat = np.concatenate((mat, I), axis = 0)
    O_indices = [] 
    
    for j in range(m):
        if np.max(abs(mat[num_rows:n, j])) < 1e-5:
            continue # If a column is 0, just keep it.
        else:
            i = num_rows
            while abs(mat[i, j]) < 1e-5:
                i += 1 
            for j_prime in np.arange(j + 1, m):
                mat[:, j_prime] -= mat[i, j_prime]/mat[i, j]*mat[:, j]
    
    # scan the matrix from left to right and remember the column that are zero with respect to row indices [num_rows:n]
    for j in range(m):
        if np.max(abs(mat[num_rows:n, j])) < 1e-5:
            O_indices.append(j)
    num_O_t0 = sum(np.array(O_indices) < num_A_t0)# dimension of Omega_{p+1} weight <= t0
    return mat[:num_rows, O_indices], mat[-m:, O_indices], num_O_t0, len(O_indices) 

def partial_column_reduction(mat, num_rows):
    """
    This function can do a column reduction with respect to certain row indices.
    parameter
    _________
    mat: a matrix. will be the omega 'boundary' matrix whose domain is omega p+1 paths and 
    codomain is regular p paths (indeed \partial(A_{p+1}), a subset of regular p paths).
    num_rows: will be the dimension of Omega paths with weight <= t0
    Output
    ------
    mat[:num_rows, indices] is the matrix from C^{t_0, t_1}_{p+1}(omega) to omega p paths with weight <= t0.
    mat[-m:, indices] is the transformation from C^{t_0, t_1}_{p+1}(omega) to Omega p+1 path with weight <= t1
    Column reduction.
    """
    mat = mat.astype('float64')
    n, m = mat.shape 
    # in this case, \partial(Omega_{p+1}) is a subset of A_p and Omega_{p+1} = A_{p+1}
    if num_rows == n:
        return mat, np.identity(m)
    I = np.identity(m)
    mat = np.concatenate((mat, I), axis = 0)
    indices = [] 
    
    for j in range(m):
        if np.max(abs(mat[num_rows:n, j])) < 1e-5:
            continue # If a column is 0, just keep it.
        else:
            i = num_rows
            while abs(mat[i, j]) < 1e-5:
                i += 1 
            for j_prime in np.arange(j + 1, m):
                mat[:, j_prime] -= mat[i, j_prime]/mat[i, j]*mat[:, j]
    
    # scan the matrix from left to right and remember the column that are zero with respect to row indices [num_rows:n]
    for j in range(m):
        if np.max(abs(mat[num_rows:n, j])) < 1e-5:
            indices.append(j)
    return mat[:num_rows, indices], mat[-m:, indices] 

def boundary1(num_nodes, edges):  
    # b_1 is the boundary matrix
    b_1 = np.zeros((num_nodes, len(edges)))
    for edge, idx_and_t in edges.items():
        b_1[edge[0], idx_and_t[0]] = -1
        b_1[edge[1], idx_and_t[0]] = 1
    return b_1

def boundary2(edges, faces):
    b_2 = np.zeros((len(edges), len(faces)))
    for face, idx_and_t in faces.items(): 
        face_face0 = (face[1], face[2])
        face_face1 = (face[0], face[2])
        face_face2 = (face[0], face[1])
        b_2[edges[face_face0][0], idx_and_t[0]] = 1
        b_2[edges[face_face1][0], idx_and_t[0]] = -1
        b_2[edges[face_face2][0], idx_and_t[0]] = 1
    return b_2

def boundary3(faces, tetras):
    b_3 = np.zeros((len(faces), len(tetras)))
    for tetra, idx_and_t in tetras.items():
        # construct faces of a 3-path. 
        tetra_face0 = (tetra[1], tetra[2], tetra[3])
        tetra_face1 = (tetra[0], tetra[2], tetra[3])
        tetra_face2 = (tetra[0], tetra[1], tetra[3])
        tetra_face3 = (tetra[0], tetra[1], tetra[2])
        b_3[faces[tetra_face0][0], idx_and_t[0]] = 1
        b_3[faces[tetra_face1][0], idx_and_t[0]] = -1
        b_3[faces[tetra_face2][0], idx_and_t[0]] = 1
        b_3[faces[tetra_face3][0], idx_and_t[0]] = -1
    return b_3

class PPL():
    def __init__(self, num_nodes, edges, p=0):
        """
        edges: eg. [[[0,1], 2.], [[1,2], 3.]]. For simplicity, [a,b] and [b,a] will not coexist.
        p: t1-t0 = p
        """
        self.num_nodes = num_nodes
        edges = sorted(edges, key=itemgetter(1))
        self.edges = edges
        self.p = p 

    def build_allowed_paths(self, t0):
        """
        0-paths are just all nodes.
        """
        t1 = t0 + self.p
        self.A1_t0, self.A2_t0, self.A3_t0, self.A1_t1, self.A2_t1, self.A3_t1 = {}, {}, {}, {}, {}, {}
        # 1-path
        for i, (edge, weight) in enumerate(self.edges):
            if weight <= t0:
                self.A1_t0[tuple(edge)] = (i, weight)
            if weight <= t1:
                self.A1_t1[tuple(edge)] = (i, weight)
        # 2-path
        tmp0 = []
        for edge0, w0 in self.edges:
            for edge1, w1 in self.edges:
                if edge0[1] == edge1[0]:
                    tmp0.append([(edge0[0], edge0[1], edge1[1]), max(w0, w1)])
        tmp0 = sorted(tmp0, key=itemgetter(1))
        for i, (path, w) in enumerate(tmp0):
            if w <= t0:
                self.A2_t0[path] = (i, w)
            self.A2_t1[path] = (i, w)
        # 3-path
        tmp0 = [] 
        for path, idx_w0 in self.A2_t1.items():
            for edge, idx_w1 in self.A1_t1.items():
                if path[2] == edge[0]:
                    tmp0.append([(path[0], path[1], path[2], edge[1]), max(idx_w0[1], idx_w1[1])])
        tmp0 = sorted(tmp0, key=itemgetter(1))
        for i, (path, w) in enumerate(tmp0):
            if w <= t0:
                self.A3_t0[path]= (i, w)
            self.A3_t1[path] = (i, w)        

    def build_regular_path(self):
        """
        construct extra regular paths that will be used in finding omega paths
        """
        tmp = len(self.A1_t1)
        self.R1 = copy.deepcopy(self.A1_t1)
        for path in self.A2_t1.keys():
            if (path[0], path[1]) not in self.R1:
                self.R1[(path[0], path[1])]=(tmp, np.inf) 
                tmp += 1
            if (path[0], path[2]) not in self.R1:
                self.R1[(path[0], path[2])]=(tmp, np.inf) 
                tmp += 1
            if (path[1], path[2]) not in self.R1:
                self.R1[(path[1], path[2])]=(tmp, np.inf) 
                tmp += 1

        tmp = len(self.A2_t1)
        self.R2 = copy.deepcopy(self.A2_t1)
        for path in self.A3_t1.keys():
            if (path[1], path[2], path[3]) not in self.R2:
                self.R2[(path[1], path[2], path[3])] = (tmp, np.inf)
                tmp += 1
            if (path[0], path[2], path[3]) not in self.R2:
                self.R2[(path[0], path[2], path[3])] = (tmp, np.inf)
                tmp += 1
            if (path[0], path[1], path[3]) not in self.R2:
                self.R2[(path[0], path[1], path[3])] = (tmp, np.inf)
                tmp += 1
            if (path[0], path[1], path[2]) not in self.R2:
                self.R2[(path[0], path[1], path[2])] = (tmp, np.inf)
                tmp += 1

    def build_matrices(self):
        """
        bulid boundary map whose domain is the set of allowed paths and codomain is the set of the 'regular' paths
        then perform partial Gaussian eliminition. 
        """
        # bulid boundary map whose domain is the set of allowed paths and codomain is the set of the 'regular' paths
        self.b_1_t1 = boundary1(self.num_nodes, self.A1_t1)
        self.b_2_t1, self.tran2, self.num_O2_t0, self.num_O2_t1 = bdryOA(boundary2(self.R1, self.A2_t1), len(self.A1_t1), len(self.A2_t0))
        self.PM_O2_t1 = self.tran2.T@self.tran2
        # bA3_t1, domain is Omega_2+1 and codomain is A_2. self.tran3, domain is Omega_3 and codomain is A_3.
        bA3_t1, self.tran3, self.num_O3_t0, self.num_O3_t1 = bdryOA(boundary3(self.R2, self.A3_t1), len(self.A2_t1), len(self.A3_t0))
        self.PM_O3_t1 = self.tran3.T@self.tran3
        if bA3_t1.size != 0:
            # solve b_3_t1
            self.b_3_t1 = np.linalg.lstsq(self.tran2, bA3_t1, rcond=None)[0]

    def L0(self): 
        return np.dot(self.b_1_t1, self.b_1_t1.T)

    def L1(self):
        self.b_1_t0 = self.b_1_t1[:, :len(self.A1_t0)]
        if self.num_O2_t1 == 0:
            return np.dot(self.b_1_t0.T, self.b_1_t0)
        if len(self.A1_t0) == len(self.A1_t1):
            return np.dot(self.b_2_t1, np.linalg.inv(self.tran2.T@self.tran2)@self.b_2_t1.T) + np.dot(self.b_1_t0.T, self.b_1_t0)
        else:
            tmp, tmpT = partial_column_reduction(self.b_2_t1, len(self.A1_t0))
        return np.dot(self.b_1_t0.T, self.b_1_t0) + np.dot(tmp, np.linalg.inv(tmpT.T@self.PM_O2_t1@tmpT)@tmp.T)

    def L2(self):
        self.b_2_t0 = self.b_2_t1[:len(self.A1_t0), :self.num_O2_t0]
        if self.num_O3_t1 == 0: # only consider Omega_1 and Omega_2
            return np.dot(np.linalg.inv(self.tran2[:, :self.num_O2_t0].T@self.tran2[:, :self.num_O2_t0])@self.b_2_t0.T, self.b_2_t0)
        if self.num_O2_t0 == self.num_O2_t1:   
            return np.dot(self.b_3_t1, np.linalg.inv(self.tran3.T@self.tran3)@self.b_3_t1.T@self.tran2.T@self.tran2) + np.dot(np.linalg.inv(self.tran2[:,:self.num_O2_t0].T@self.tran2[:,:self.num_O2_t0])@self.b_2_t0.T, self.b_2_t0)
        else:
            tmp, tmpT = partial_column_reduction(self.b_3_t1, self.num_O2_t0)
        return  np.dot(np.linalg.inv(self.tran2[:,:self.num_O2_t0].T@self.tran2[:,:self.num_O2_t0])@self.b_2_t0.T, self.b_2_t0) + np.dot(tmp, np.linalg.inv(tmpT.T@self.PM_O3_t1@tmpT)@tmp.T@self.PM_O2_t1[:self.num_O2_t0, :self.num_O2_t0])

