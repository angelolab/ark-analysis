from cython cimport cdivision, boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand
from libc.string cimport memset

ctypedef np.uint16_t DTYPE_t
ctypedef np.uint8_t UINT8_t

@cdivision(True) # Ignore modulo/divide by zero warning
cdef inline void _c_permutation(Py_ssize_t* arr, const Py_ssize_t size) nogil:
    cdef Py_ssize_t i, j, temp

    for i in range(size-1, -1, -1):
        j = rand() % (i + 1)
        temp = arr[i]
        arr[i] = j
        arr[j] = temp

@cdivision(True) # Ignore modulo/divide by zero warning
cdef inline void _init_flag_table(UINT8_t* flags, const Py_ssize_t* perm,
                                  const Py_ssize_t size) nogil:
    cdef Py_ssize_t idx
    for idx in range(size):
        flags[perm[idx]] = 1


@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True)    # Ignore modulo/divide by zero warning
cdef inline void _list_accum(DTYPE_t[:] close_num_rand_view,
                             const DTYPE_t[:, :] dist_mat_bin, Py_ssize_t* rand_rows,
                             Py_ssize_t* rand_cols, Py_ssize_t num_choices, int m1n, int m2n,
                             int bootstrap_num) nogil:
    cdef DTYPE_t accum
    cdef Py_ssize_t m1_label, m2_label
    for r in range(bootstrap_num):
        accum = 0
        _c_permutation(rand_rows, num_choices)
        _c_permutation(rand_cols, num_choices)
        for m1_label in rand_rows[:m1n]:
            for m2_label in rand_cols[:m2n]:
                accum += dist_mat_bin[m1_label, m2_label]
        close_num_rand_view[r] = accum
        

@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True)    # Ignore modulo/divide by zero warning
cdef inline void _dict_accum(DTYPE_t[:] close_num_rand_view,
                             const DTYPE_t[:] cols_in_row_flat, const DTYPE_t[:] row_indicies,
                             Py_ssize_t* rand_rows, Py_ssize_t* rand_cols,
                             UINT8_t* rand_cols_flags, Py_ssize_t num_choices, int m1n, int m2n,
                             int bootstrap_num) nogil:
    cdef DTYPE_t accum, flat_start, flat_end, m1_label, m2_label, m2_idx
    for r in range(bootstrap_num):
        accum = 0
        _c_permutation(rand_rows, num_choices)
        _c_permutation(rand_cols, num_choices)
        memset(rand_cols_flags, 0, num_choices * sizeof(UINT8_t))
        _init_flag_table(rand_cols_flags, rand_cols, m2n)
        for m1_label in rand_rows[:m1n]:
            flat_start = row_indicies[m1_label]
            flat_end = row_indicies[m1_label + 1]
            for m2_idx in range(flat_start, flat_end):
                m2_label = cols_in_row_flat[m2_idx]
                accum += rand_cols_flags[m2_label]
        close_num_rand_view[r] = accum


@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True)    # Ignore modulo/divide by zero warning
cdef _compute_close_num_rand(DTYPE_t[:, :] dist_mat_bin, DTYPE_t[:] cols_in_row_flat,
                            DTYPE_t[:] row_indicies, DTYPE_t[:] marker_nums,
                            int bootstrap_num):

    cdef Py_ssize_t num_markers = marker_nums.shape[0]
    cdef Py_ssize_t num_choices = row_indicies.shape[0] - 1

    close_num_rand = np.zeros((num_markers, num_markers, bootstrap_num), dtype=np.uint16)
    cdef DTYPE_t[:, :, :] close_num_rand_view = close_num_rand
    
    # allocate marker_label randomization containers
    cdef Py_ssize_t* rand_rows = <Py_ssize_t*> PyMem_Malloc(num_choices * sizeof(Py_ssize_t))
    cdef Py_ssize_t* rand_cols = <Py_ssize_t*> PyMem_Malloc(num_choices * sizeof(Py_ssize_t))
    cdef UINT8_t* rand_cols_flags = <UINT8_t*> PyMem_Malloc(num_choices * sizeof(UINT8_t))
    if not rand_rows or not rand_cols or not rand_cols_flags:
        raise MemoryError()

    # allocate 'm2n < avg_rowsize' memory
    cdef UINT8_t* m2n_small = <UINT8_t*> PyMem_Malloc(num_markers * sizeof(UINT8_t))

    # initialize marker_label containers and m2n_small
    cdef int mn
    cdef DTYPE_t avg_rowsize
    for mn in range(num_choices):
        rand_rows[mn] = mn
        rand_cols[mn] = mn
        avg_rowsize = row_indicies[mn + 1] - row_indicies[mn]

    avg_rowsize /= num_choices
    for mn in range(num_markers):
        m2n_small[mn] = (marker_nums[mn] < avg_rowsize)

    cdef int j, k, r, m1n, m2n
    cdef Py_ssize_t m1_label, m2_label

    cdef DTYPE_t accum, flat_start, flat_end, m2_idx
    for j in range(num_markers):
        m1n = marker_nums[j]
        for k in range(j, num_markers):
            m2n = marker_nums[k]
            if m2n_small[k]:
                _list_accum(close_num_rand_view[j, k, :], dist_mat_bin, rand_rows, rand_cols,
                            num_choices, m1n, m2n, bootstrap_num)
            else:
                _dict_accum(close_num_rand_view[j, k, :], cols_in_row_flat, row_indicies,
                            rand_rows, rand_cols, rand_cols_flags, num_choices, m1n, m2n,
                            bootstrap_num)
            close_num_rand_view[k, j, :] = close_num_rand_view[j, k, :]

    PyMem_Free(rand_rows)
    PyMem_Free(rand_cols)
    PyMem_Free(rand_cols_flags)
    PyMem_Free(m2n_small)

    return close_num_rand

def compute_close_num_rand(DTYPE_t[:, :] dist_mat_bin, DTYPE_t[:] cols_in_row_flat,
                           DTYPE_t[:] row_indicies, DTYPE_t[:] marker_nums, int bootstrap_num):
    return _compute_close_num_rand(dist_mat_bin, cols_in_row_flat, row_indicies, marker_nums,
                                   bootstrap_num)

'''
//     cdef np.ndarray marker1_labels_rand = [np.permutation(num_choices)[:m1n]]
//     cdef np.ndarray marker2_labels_rand = [np.permutation(num_choices)[:m2n]]

// for j, m1n in enumerate(marker_nums):
//     for k, m2n in enumerate(marker_nums[j:], j):
//         for r in range(bootstrap_num):
//             # Select same amount of random cell labels as positive ones in close_num
//             marker1_labels_rand = np.random.choice(a=choice_ar, size=m1n, replace=False)
//             marker2_labels_rand = np.random.choice(a=choice_ar, size=m2n, replace=False)

//             # Record the number of interactions and store in close_num_rand in the index
//             # corresponding to both markers, for every permutation
//             close_num_rand[j, k, r] = \
//                 np.sum(dist_mat_bin[np.ix_(marker1_labels_rand, marker2_labels_rand)])

//         # System should be symetric
//         close_num_rand[k, j, :] = close_num_rand[j, k, :]
// return close_num_rand
'''