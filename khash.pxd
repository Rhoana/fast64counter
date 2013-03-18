from numpy cimport int64_t, int32_t, uint32_t

cdef extern from "khash.h":
    ctypedef uint32_t khint_t
    ctypedef khint_t khiter_t
    ctypedef struct int64pair_t:
        int64_t a
        int64_t b

    ctypedef struct kh_int64_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int64_t *keys
        size_t *vals

    ctypedef struct kh_int64pair_t:
        khint_t n_buckets, size, n_occupied, upper_bound
        uint32_t *flags
        int64pair_t *keys
        size_t *vals

    inline kh_int64_t* kh_init_int64()
    inline void kh_destroy_int64(kh_int64_t*)
    inline void kh_clear_int64(kh_int64_t*)
    inline khint_t kh_get_int64(kh_int64_t*, int64_t)
    inline void kh_resize_int64(kh_int64_t*, khint_t)
    inline khint_t kh_put_int64(kh_int64_t*, int64_t, int*)
    inline void kh_del_int64(kh_int64_t*, khint_t)
    bint kh_exist_int64(kh_int64_t*, khiter_t)

    inline kh_int64pair_t* kh_init_int64pair()
    inline void kh_destroy_int64pair(kh_int64pair_t*)
    inline void kh_clear_int64pair(kh_int64pair_t*)
    inline khint_t kh_get_int64pair(kh_int64pair_t*, int64pair_t)
    inline void kh_resize_int64pair(kh_int64pair_t*, khint_t)
    inline khint_t kh_put_int64pair(kh_int64pair_t*, int64pair_t, int*)
    inline void kh_del_int64pair(kh_int64pair_t*, khint_t)
    bint kh_exist_int64pair(kh_int64pair_t*, khiter_t)
