from khash cimport *
from numpy cimport *

import numpy as np

cimport cython
cimport numpy as cnp

cnp.import_array()
cnp.import_ufunc()

ctypedef fused f_int64:
    int64_t
    uint64_t

cdef class ValueCountInt64:
    cdef kh_int64_t *table

    def __cinit__(self):
        self.table = kh_init_int64()

    def __dealloc__(self):
        kh_destroy_int64(self.table)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef add_values(self, ndarray[f_int64] values):
        cdef:
            int64_t val
            Py_ssize_t i, k, n = len(values)
            int ret = 0
            ndarray[int64_t] _values

        _values = values.view(dtype=np.int64)
        for i in range(n):
            val = values[i]
            k = kh_get_int64(self.table, val)
            if k != self.table.n_buckets:
                self.table.vals[k] += 1
            else:
                k = kh_put_int64(self.table, val, &ret)
                self.table.vals[k] = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef add_values_32(self, ndarray[int32_t] values):
        cdef:
            int64_t val
            Py_ssize_t i, k, n = len(values)
            int ret = 0

        for i in range(n):
            val = <int64_t> (values[i])
            k = kh_get_int64(self.table, val)
            if k != self.table.n_buckets:
                self.table.vals[k] += 1
            else:
                k = kh_put_int64(self.table, val, &ret)
                self.table.vals[k] = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef add_values_pair32(self, ndarray[int32_t] vhi, ndarray[int32_t] vlo):
        cdef:
            int64_t val
            Py_ssize_t i, k, n = len(vlo)
            int ret = 0

        assert len(vlo) == len(vhi)
        for i in range(n):
            val = ((<int64_t> vhi[i]) << 32) + (<int64_t> vlo[i])
            k = kh_get_int64(self.table, val)
            if k != self.table.n_buckets:
                self.table.vals[k] += 1
            else:
                k = kh_put_int64(self.table, val, &ret)
                self.table.vals[k] = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef get_counts(self):
        cdef:
            Py_ssize_t k, i = 0
            ndarray[int64_t] result_keys, result_counts
        result_keys = np.empty(self.table.n_occupied, dtype=np.int64)
        result_counts = np.zeros(self.table.n_occupied, dtype=np.int64)
        for k in range(self.table.n_buckets):
            if kh_exist_int64(self.table, k):
                result_keys[i] = self.table.keys[k]
                result_counts[i] = self.table.vals[k]
                i += 1
        return result_keys, result_counts

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef get_counts_pair32(self):
        cdef:
            Py_ssize_t k, i = 0
            int64_t result_key
            ndarray[int32_t] result_keys_hi, result_keys_lo
            ndarray[int64_t] result_counts
        result_keys_hi = np.empty(self.table.n_occupied, dtype=np.int32)
        result_keys_lo = np.empty(self.table.n_occupied, dtype=np.int32)
        result_counts = np.zeros(self.table.n_occupied, dtype=np.int64)
        for k in range(self.table.n_buckets):
            if kh_exist_int64(self.table, k):
                result_key = self.table.keys[k]
                result_keys_hi[i] = (result_key >> 32)
                result_keys_lo[i] = (result_key & <int32_t> 0xffffffff)
                result_counts[i] = self.table.vals[k]
                i += 1
        return result_keys_hi, result_keys_lo, result_counts

cdef class ValueCountPair64:
    cdef kh_int64pair_t *table

    def __cinit__(self):
        self.table = kh_init_int64pair()

    def __dealloc__(self):
        kh_destroy_int64pair(self.table)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef add_values_pair(self, ndarray[f_int64] first, ndarray[f_int64] second):
        cdef:
            int64pair_t val
            Py_ssize_t i, k, n = len(first)
            int ret = 0
            ndarray[int64_t] _first
            ndarray[int64_t] _second

        assert len(first) == len(second)
        _first = first.view(dtype=np.int64)
        _second = second.view(dtype=np.int64)

        for i in range(n):
            val.a = _first[i]
            val.b = _second[i]
            k = kh_get_int64pair(self.table, val)
            if k != self.table.n_buckets:
                self.table.vals[k] += 1
            else:
                k = kh_put_int64pair(self.table, val, &ret)
                self.table.vals[k] = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef add_values_pair32(self, ndarray[int32_t] first, ndarray[int32_t] second):
        cdef:
            int64pair_t val
            Py_ssize_t i, k, n = len(first)
            int ret = 0

        assert len(first) == len(second)
        for i in range(n):
            val.a = first[i]
            val.b = second[i]
            k = kh_get_int64pair(self.table, val)
            if k != self.table.n_buckets:
                self.table.vals[k] += 1
            else:
                k = kh_put_int64pair(self.table, val, &ret)
                self.table.vals[k] = 1

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef get_counts(self):
        cdef:
            Py_ssize_t k, i = 0
            ndarray[int64_t] result_first_keys, result_second_keys, result_counts
        result_first_keys = np.empty(self.table.n_occupied, dtype=np.int64)
        result_second_keys = np.empty(self.table.n_occupied, dtype=np.int64)
        result_counts = np.zeros(self.table.n_occupied, dtype=np.int64)
        for k in range(self.table.n_buckets):
            if kh_exist_int64pair(self.table, k):
                result_first_keys[i] = self.table.keys[k].a
                result_second_keys[i] = self.table.keys[k].b
                result_counts[i] = self.table.vals[k]
                i += 1
        return result_first_keys, result_second_keys, result_counts
