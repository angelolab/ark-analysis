from cython cimport cdivision, boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free

from numpy import zeros, uint16
from numpy cimport uint16_t

from libc.stdlib cimport rand

ctypedef uint16_t DTYPE_t

@cdivision(True) # Ignore modulo/divide by zero warning
cdef _c_permutation(Py_ssize_t* array, int size):
    cdef int i, j

    for i in range(size-1, -1, -1):
        j = rand() % (i + 1)
        array[i] = j
        array[j] = i


@boundscheck(False)  # Deactivate bounds checking
@wraparound(False)   # Deactivate negative indexing.
def compute_close_num_rand(DTYPE_t[:, :] dist_mat_bin, DTYPE_t[:] marker_nums,
                           DTYPE_t[:] choice_ar, int bootstrap_num):

    cdef Py_ssize_t num_markers = marker_nums.shape[0]
    cdef Py_ssize_t num_choices = choice_ar.shape[0]

    # check marker num sizes
    cdef int mn
    for mn in range(num_markers):
        assert num_choices >= marker_nums[mn]

    close_num_rand = zeros((num_markers, num_markers, bootstrap_num), dtype=uint16)
    cdef DTYPE_t[:, :, :] close_num_rand_view = close_num_rand
    
    cdef Py_ssize_t* marker1_labels = <Py_ssize_t*> PyMem_Malloc(num_choices * sizeof(Py_ssize_t))
    cdef Py_ssize_t* marker2_labels = <Py_ssize_t*> PyMem_Malloc(num_choices * sizeof(Py_ssize_t))
    if not marker1_labels or not marker2_labels:
        raise MemoryError()

    cdef int j, k, r, m1n, m2n, label_i, label_j, m1_label, m2_label

    for j in range(num_markers):
        m1n = marker_nums[j]
        for k in range(j, num_markers):
            m2n = marker_nums[k]
            for r in range(bootstrap_num):
                _c_permutation(marker1_labels, m1n)
                _c_permutation(marker2_labels, m2n)
                #marker1_labels = np.permutation(num_choices)
                #marker2_labels = np.permutation(num_choices)
                for label_i in range(m1n):
                    m1_label = marker1_labels[label_i]
                    for label_j in range(m2n):
                        m2_label = marker2_labels[label_j]
                        close_num_rand_view[j, k, r] += dist_mat_bin[m1_label, m2_label]
            close_num_rand_view[k, j, :] = close_num_rand_view[j, k, :]

    PyMem_Free(marker1_labels)
    PyMem_Free(marker2_labels)

    return close_num_rand

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