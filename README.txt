A fast 64-bit python counter (like collections.Counter, but only for numpy
arrays of 64 bit integers or pairs of 32-bit integers), built as a wrapper
around the klib hashtable implementation[1], with code borrowed from pandas[2].

The khash code is modified to remove the ability to delete an entry,
since this code is only used for counting, which gives ~10% faster
performance.

[1] https://github.com/attractivechaos/klib
[2] https://github.com/pydata/pandas
