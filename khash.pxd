from numpy cimport int64_t, int32_t, uint32_t

cdef extern from "khash_python.h":
    ctypedef uint32_t khint_t
    ctypedef khint_t khiter_t

    ctypedef struct kh_int64_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int64_t *keys
        size_t *vals

    inline kh_int64_t* kh_init_int64()
    inline void kh_destroy_int64(kh_int64_t*)
    inline void kh_clear_int64(kh_int64_t*)
    inline khint_t kh_get_int64(kh_int64_t*, int64_t)
    inline void kh_resize_int64(kh_int64_t*, khint_t)
    inline khint_t kh_put_int64(kh_int64_t*, int64_t, int*)
    inline void kh_del_int64(kh_int64_t*, khint_t)

    bint kh_exist_int64(kh_int64_t*, khiter_t)
