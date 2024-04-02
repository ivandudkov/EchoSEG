import abc
import struct
from collections import namedtuple


import timeit

# Tuple with names of low-level elements
_BinTypes = namedtuple("BinTypes", 
                      [
                          "c8",  # char
                          "i8",  # signed,
                          "u8",  # unsigned, unsigned char
                          "i16",  # int 16bit
                          "u16",  # unsigned int, 16bit
                          "i32",  # signed int, 32bit
                          "u32",  # unsigned int, 32 bit
                          "i64",  # signed int, 64 bit
                          "u64",  # unsigned int, 64 bit
                          "f32",  # float, 32 bit
                          "f64"  # double, float 64 bit
                          ]
                      )

binT = _BinTypes(*_BinTypes._fields)

def elemBlock(name: str, fmt: str, count=1):
    return(name, (fmt, count))

#  for C types, numpy types, count
map_size_to_fmt = dict(
    (
        (binT.c8, ("c", "B", 1)),
        (binT.i8, ("b", "b", 1)),
        (binT.u8, ("B", "u1", 1)),
        (binT.i16, ("h", "i2", 2)),
        (binT.u16, ("H", "u2", 2)),
        (binT.i32, ("i", "i4", 4)),
        (binT.u32, ("I", "u4", 4)),
        (binT.i64, ("q", "i8", 8)),
        (binT.u64, ("Q", "u8", 8)),
        (binT.f32, ("f", "f4", 4)),
        (binT.f64, ("d", "f8", 8)),
    )
)


class Datagram(metaclass=abc.ABCMeta):
    """
    Datablock reader
    """
    
    _byte_order_fmt = "<"
    
    def __init__(self, bin_element):
        self._sizes = ''
        self._names = ''
        self.struct = ''
        self.np_types = ''
        
    @property
    def size(self):
        return self._sizes