from cpython.mem cimport PyMem_Free, PyMem_Malloc
from cython cimport boundscheck, cdivision, wraparound

import numpy as np

cimport numpy as np
from libc.stdlib cimport rand
from libc.string cimport memset

ctypedef np.uint64_t MAXINDEX_t
ctypedef np.uint16_t DTYPE_t
ctypedef np.uint8_t UINT8_t

@cdivision(True) # Ignore modulo/divide by zero warning
cdef inline void _c_permutation(Py_ssize_t* arr, const Py_ssize_t size) nogil:
    """ Randomly permutes the provided arr

    Very common implementation

    Args:
        arr (Py_ssize_t*):
            array to be permuted/shuffled
        size (Py_ssize_t):
            size of array to be shuffled
    """
    cdef Py_ssize_t i, j, temp

    for i in range(size-1, 0, -1):
        j = rand() % (i + 1)
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp


@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True) # Ignore modulo/divide by zero warning
cdef inline void _init_flag_table(UINT8_t* flags, const Py_ssize_t* perm,
                                  const Py_ssize_t size) nogil:
    """ Initializes lookup table according to provided permutation

    Args:
        flags (uint8_t*):
            pointer to mutable randomized column lookup table memory block
        perm (Py_ssize_t*):
            pointer to immutable permutation memory block
        size (Py_ssize_t):
            size of the lookup table/permutation
    """
    cdef Py_ssize_t idx
    for idx in range(size):
        flags[perm[idx]] = 1


@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True)    # Ignore modulo/divide by zero warning
cdef inline void _list_accum(DTYPE_t[:] close_num_rand_view,
                             const DTYPE_t[:, :] dist_mat_bin, MAXINDEX_t[:] pos_labels,
                             Py_ssize_t* rand_cols, Py_ssize_t num_choices, int m1n, int m2n,
                             int bootstrap_num):
    """ List based accumulation for small secondary marker size

    Accumulation loops over a randomized list of columnns within a randomized list of rows.  The
    binarized distance matrix is directly indexed via these cols/rows and accumulated to the
    result.

    This is faster when the average number of positive columns in a row is more than the number of
    random columns to check.

    Args:
        close_num_rand_view (np.ndarray[np.uint16]):
            typed memory view of the close_num_rand_view datastructure
        dist_mat_bin (np.ndarray[np.uint16]):
            binarized distance matrix
        pos_labels (np.ndarray[np.uint64]):
            marker indices within dist_mat_bin    
        rand_cols (Py_ssize_t*):
            pointer to mutable column randomization memory block
        num_choices (Py_ssize_t):
            number of choices for permutation generation
        m1n (int):
            number of rows to select
        m2n (int):
            number of columns to select
        bootstrap_num (int):
            number of bootstrap iterations

    """
    cdef DTYPE_t accum
    cdef Py_ssize_t m1_label, m2_label

    if not m1n or not m2n:
        return

    for r in range(bootstrap_num):
        accum = 0
        _c_permutation(rand_cols, num_choices)
        for m1_label in pos_labels:
            for m2_label in rand_cols[:m2n]:
                accum += dist_mat_bin[m1_label, m2_label]
        close_num_rand_view[r] = accum

@boundscheck(False) # Deactivate bounds checking
@wraparound(False)  # Deactivate negative indexing
@cdivision(True)    # Ignore modulo/divide by zero warning
cdef inline void _dict_accum(DTYPE_t[:] close_num_rand_view,
                             const DTYPE_t[:] cols_in_row_flat, const MAXINDEX_t[:] row_indicies,
                             const MAXINDEX_t[:] pos_labels, Py_ssize_t* rand_cols,
                             UINT8_t* rand_cols_flags, Py_ssize_t num_choices, int m1n, int m2n,
                             int bootstrap_num):
    """ Dictionary based accumulation for large secondary marker size

    Accumulation still loops over a random list of rows, but instead of directly indexing the
    binarized distance matrix over a random list of columns, the known positive columns within each
    row are checked against a randomized 'positive column' lookup table (dictionary).
    
    This is faster when the average number of positive columns in a row is significantly less than
    the number of random columns to check (hence why sorting marker column order low to high is
    encouraged...)

    Args:
        close_num_rand_view (np.ndarray[np.uint16]):
            typed memory view of the close_num_rand_view datastructure
        cols_in_row_flat (np.ndarray[np.uint16]):
            flattened list-of-lists representation of binarized distance matrix
        row_indicies (np.ndarray[np.uint64]):
            'deflattening' index array for `cols_in_row_flat`
        pos_labels (np.ndarray[np.uint64]):
            marker indices within dist_mat_bin    
        rand_cols (Py_ssize_t*):
            pointer to mutable column randomization memory block
        rand_cols_flags (uint8_t*):
            pointer to mutable randomized column lookup table memory block
        num_choices (Py_ssize_t):
            number of choices for permutation generation
        m1n (int):
            number of rows to select
        m2n (int):
            number of columns to select
        bootstrap_num (int):
            number of bootstrap iterations
    """
    cdef DTYPE_t accum
    cdef MAXINDEX_t flat_start, flat_end, m2_idx
    cdef Py_ssize_t m1_label, m2_label

    if not m1n or not m2n:
        return

    for r in range(bootstrap_num):   
        accum = 0
        _c_permutation(rand_cols, num_choices)
        memset(rand_cols_flags, 0, num_choices * sizeof(UINT8_t))
        _init_flag_table(rand_cols_flags, rand_cols, m2n)
        for m1_label in pos_labels: 
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
                             MAXINDEX_t[:] row_indicies, DTYPE_t[:] marker_nums,
                             dict pos_labels, int bootstrap_num):
    """ Cython implementation of the spatial enrichment bootstrapper

    Args:
        dist_mat_bin (np.ndarray[np.uint16]):
            binarized distance matrix.  Dtype is not np.uint8 as to avoid overflows on opperations
            involving this matrix.
        cols_in_row_flat (np.ndarray[np.uint16]):
            flattened list-of-lists sparse matrix representation of `dist_mat_bin`.
        row_indicies (np.ndarray[np.uint64]):
            mapping from row to index of `cols_in_row_flat` containing list of positive columns for
            that row.
        marker_nums (np.ndarray[np.uint16]):
            number of hits for each marker
        pos_labels (dict):
            marker indices within dist_mat_bin
        bootstrap_num (int):
            number of bootstrapping iterations to perform

    Returns:
        np.ndarray[np.uint16]:
            3d array containing the enrichment matrix for each bootstrap iteration
    """

    # premptively type cast to native signed size type
    cdef Py_ssize_t num_markers = marker_nums.shape[0]
    cdef Py_ssize_t num_choices = dist_mat_bin.shape[0]

    # pre-allocate space for close_num_rand and create memory view for fast access
    close_num_rand = np.zeros((num_markers, num_markers, bootstrap_num), dtype=np.uint16)
    cdef DTYPE_t[:, :, :] close_num_rand_view = close_num_rand
    
    # allocate marker_label randomization containers
    cdef Py_ssize_t* rand_cols = <Py_ssize_t*> PyMem_Malloc(num_choices * sizeof(Py_ssize_t))
    cdef UINT8_t* rand_cols_flags = <UINT8_t*> PyMem_Malloc(num_choices * sizeof(UINT8_t))
    if not rand_cols or not rand_cols_flags:
        raise MemoryError()

    # allocate 'm2n < avg_rowsize' memory
    # when m2n_small is true, traditional accumulation is used
    # whne m2n_small is false, dictionary indexed accumulation is faster
    cdef UINT8_t* m2n_small = <UINT8_t*> PyMem_Malloc(num_markers * sizeof(UINT8_t))

    # initialize marker_label containers and m2n_small
    cdef int mn
    cdef DTYPE_t avg_rowsize
    for mn in range(num_choices):
        rand_cols[mn] = mn
        avg_rowsize += row_indicies[mn + 1] - row_indicies[mn]

    avg_rowsize /= num_choices
    for mn in range(num_markers):
        m2n_small[mn] = (marker_nums[mn] < avg_rowsize)

    cdef int j, k, row_ind, m1n, m2n, swap_buf

    # start main bootstrapping loop
    # TODO: split randomization from row/column choice (recovers get m1n < m2n speed boost)
    for j in range(num_markers):
        m1n = marker_nums[j]
        for k in range(num_markers):
            m2n = marker_nums[k]
            if m2n_small[k]:
                _list_accum(close_num_rand_view[j, k, :], dist_mat_bin,
                            pos_labels[j], rand_cols, num_choices, m1n, m2n,
                            bootstrap_num)
            else:
                _dict_accum(close_num_rand_view[j, k, :], cols_in_row_flat, row_indicies,
                            pos_labels[j], rand_cols, rand_cols_flags, num_choices, m1n,
                            m2n, bootstrap_num)

    # free used memeory
    PyMem_Free(rand_cols)
    PyMem_Free(rand_cols_flags)
    PyMem_Free(m2n_small)

    return close_num_rand

def compute_close_num_rand(DTYPE_t[:, :] dist_mat_bin, DTYPE_t[:] cols_in_row_flat,
                           MAXINDEX_t[:] row_indicies, DTYPE_t[:] marker_nums,
                           dict pos_labels, int bootstrap_num):
    """ Python wrapper function for the cython implementation of the spatial enrichment
        bootstrapper

    Args:
        dist_mat_bin (np.ndarray[np.uint16]):
            binarized distance matrix.  Dtype is not np.uint8 as to avoid overflows on opperations
            involving this matrix.  This should be sorted in increasing marker size order for
            optimal speedups.
        cols_in_row_flat (np.ndarray[np.uint16]):
            flattened list-of-lists sparse matrix representation of `dist_mat_bin`.
        row_indicies (np.ndarray[np.uint64]):
            mapping from row to index of `cols_in_row_flat` containing list of positive columns for
            that row.
        marker_nums (np.ndarray[np.uint16]):
            number of hits for each marker
        pos_labels (dict):
            marker indices within dist_mat_bin
        bootstrap_num (int):
            number of bootstrapping iterations to perform

    Returns:
        np.ndarray[np.uint16]:
            3d array containing the enrichment matrix for each bootstrap iteration
    """
    return _compute_close_num_rand(dist_mat_bin, cols_in_row_flat, row_indicies, marker_nums,
                                   pos_labels, bootstrap_num)