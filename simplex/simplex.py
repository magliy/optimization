import numpy as np


epsilon = 1e-14


def zero_pos_indices_split(x):
    N_zero = []
    N_pos = []
    
    for index in range(len(x)):
        if x[index] > epsilon:
            N_pos.append(index)
        else:
            N_zero.append(index)

    return N_zero, N_pos


def is_full_rank_matrix(A):
    return min(A.shape) == matrix_rank(A)


def matrix_rank(A):
    return np.linalg.matrix_rank(A)


def matrix_inverse(A):
    assert A.shape[0] == A.shape[1]
    
    return np.linalg.inv(A)


def find_basis(A, N_zero, N_pos):
    
    if len(N_pos) == A.shape[0]:
        return N_pos, N_zero
    
    N_k = N_pos.copy()
    curr_matrix_rank = len(N_pos)
    zeros_idx = 0
    
    while zeros_idx < len(N_zero):
        
        while zeros_idx < len(N_zero):
            
            potential_N_k = N_k+[N_zero[zeros_idx]]
            zeros_idx += 1

            new_rank = matrix_rank(A[:, potential_N_k])
            
            if new_rank > curr_matrix_rank:
                N_k = sorted(potential_N_k)
                curr_matrix_rank += 1
                break
        else:
            break
        
        if curr_matrix_rank == A.shape[0]:
            L_k = [i for i in range(A.shape[1]) if i not in N_k]
            return N_k, L_k
    
    raise ValueError('Not a full rank matrix recieved')

    
def build_d_L_k(A, B, c, N_k, L_k):
    return c[L_k] - c[N_k].dot(B.dot(A[:, L_k]))

    
def first_index(x, f):
    for i in range(len(x)):
        if f(x[i]):
            return i
    return None


def build_u_k(A, B, N_k, j_k):
    u = np.zeros(A.shape[1])
    u[N_k] = B.dot(A[:, j_k])
    u[j_k] = -1
    
    return u


def compute_theta(x, u, N_k):
    return min(x[i]/u[i] for i in N_k if u[i] > epsilon)


def change_basis(A, N_pos, N_k, L_k, N_k_diff_N_pos, j_k):

    while True:
        i_k = np.random.choice(N_k_diff_N_pos)
        N_k_copy = [i for i in N_k if i != i_k]
        N_k_copy.append(j_k)
        if matrix_rank(A[:, N_k_copy]) == A.shape[0]:
            break

    N_k.remove(i_k)
    N_k.append(j_k)
    L_k.remove(j_k)
    L_k.append(i_k)

    return N_k, L_k


def phase_one(A, b):
    sign_correction_matrix = np.eye(A.shape[0])
    
    for i in range(len(b)):
        if b[i] < -epsilon:
            sign_correction_matrix[i, i] *= -1
            
    p1_A = np.concatenate(
        (sign_correction_matrix.dot(A), np.eye(A.shape[0])),
        axis=1
    )
    p1_b = sign_correction_matrix.dot(b)
    
    y0 = np.concatenate((np.zeros(A.shape[1]), p1_b), axis=0)

    p1_c = np.array([0] * A.shape[1] + [1] * A.shape[0])

    p1_y = simplex(p1_A, p1_b, p1_c, y0)['x']

    if any(y[A.shape[1]:] > epsilon):
        raise ValueError('The prolbem is infeasible')

    return p1_y[:A.shape[1]]
        

def simplex(A, b, c, x0=None):
    
    assert A.shape[0] <= A.shape[1] and is_full_rank_matrix(A), \
           'bad constraint matrix'

    if x0 is None:
        x0 = phase_one(A, b)
    
    x_k = x0
    
    N_zero, N_pos = zero_pos_indices_split(x_k)
    N_k, L_k = find_basis(A, N_zero, N_pos)
    
    while True:
        B = matrix_inverse(A[:, N_k])

        d_L_k = build_d_L_k(A, B, c, N_k, L_k)

        '''
        change first to min
        '''
        fn = first_index(d_L_k, lambda x: x < -epsilon)
        if fn is None:
            return {'x': x_k, 'y': c[N_k].dot(B)}
        j_k = L_k[fn]

        u_k = build_u_k(A, B, N_k, j_k)

        if all(u_k[N_k] <= epsilon):
            raise ValueError('The function is unbounded')
            
        N_k_diff_N_pos = sorted(set(N_k).difference(set(N_pos)))

        first_pos = first_index(u_k[N_k_diff_N_pos], lambda x: x > epsilon)
            
        if len(N_pos) < A.shape[0] and first_pos is not None:
            N_k, L_k = change_basis(A, N_pos, N_k, L_k, N_k_diff_N_pos, j_k)
            continue
            
        theta = compute_theta(x_k, u_k, N_k)
        
        x_k = x_k - theta * u_k
            
        N_zero, N_pos = zero_pos_indices_split(x_k)
        N_k, L_k = find_basis(A, N_zero, N_pos)
