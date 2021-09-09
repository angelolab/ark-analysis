from cython cimport cdivision, boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free

import numpy as np
cimport numpy as np

from libc.stdlib cimport rand

ctypedef np.uint16_t DTYPE_t

@cdivision(True) # Ignore modulo/divide by zero warning
cdef inline void _c_permutation(Py_ssize_t* arr, const Py_ssize_t size):
    cdef Py_ssize_t i, j, temp

    for i in range(size-1, -1, -1):
        j = rand() % (i + 1)
        temp = arr[i]
        arr[i] = j
        arr[j] = temp


@boundscheck(False)  # Deactivate bounds checking
@wraparound(False)   # Deactivate negative indexing.
cdef np.ndarray _compute_close_num_rand(DTYPE_t[:, ::1] dist_mat_bin, DTYPE_t[:] marker_nums,
                           DTYPE_t[:] choice_ar, int bootstrap_num):

    cdef Py_ssize_t num_markers = marker_nums.shape[0]
    cdef Py_ssize_t num_choices = choice_ar.shape[0]

    # check marker num sizes
    cdef int mn
    for mn in range(num_markers):
        assert num_choices >= marker_nums[mn]

    close_num_rand = np.zeros((num_markers, num_markers, bootstrap_num), dtype=np.uint16)
    cdef DTYPE_t[:, :, :] close_num_rand_view = close_num_rand
    
    # allocate marker_label randomization containers
    cdef Py_ssize_t* marker1_labels = <Py_ssize_t*> PyMem_Malloc(num_choices * sizeof(Py_ssize_t))
    cdef Py_ssize_t* marker2_labels = <Py_ssize_t*> PyMem_Malloc(num_choices * sizeof(Py_ssize_t))
    if not marker1_labels or not marker2_labels:
        raise MemoryError()

    # initialize marker_label containers
    for mn in range(num_choices):
        marker1_labels[mn] = mn
        marker2_labels[mn] = mn

    cdef int j, k, r, m1n, m2n
    cdef Py_ssize_t m1_label, m2_label

    cdef DTYPE_t accum
    for j in range(num_markers):
        m1n = marker_nums[j]
        for k in range(j, num_markers):
            m2n = marker_nums[k]
            for r in range(bootstrap_num):
                accum = 0
                _c_permutation(marker1_labels, num_choices)
                _c_permutation(marker2_labels, num_choices)
                for m1_label in marker1_labels[:m1n]:
                    for m2_label in marker2_labels[:m2n]:
                        accum += dist_mat_bin[m1_label, m2_label]
                close_num_rand_view[j, k, r] = accum
                close_num_rand_view[k, j, r] = accum

    PyMem_Free(marker1_labels)
    PyMem_Free(marker2_labels)

    return close_num_rand

def compute_close_num_rand(DTYPE_t[:, ::1] dist_mat_bin, DTYPE_t[:] marker_nums,
                           DTYPE_t[:] choice_ar, int bootstrap_num):
    return _compute_close_num_rand(dist_mat_bin, marker_nums, choice_ar, bootstrap_num)

''''
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