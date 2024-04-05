import abc
import io
import struct
from collections import namedtuple, defaultdict

import numpy as np
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

#  for C types, numpy types, bytes
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
    Reads fixed-size blocks of structured binary data, according to a specified format
    """
    
    _byte_order_fmt = "<"
    
    def __init__(self, elements):
        self._sizes = self._util_take_sizes(elements)
        self._names = self._util_take_names(elements)
        self._struct = self._util_create_struct(self._sizes)
        self._np_types = self._util_create_np_types(self._names, self._sizes)
        
    @property
    def size(self):
        return self._sizes
    
    @property
    def numpy_types(self):
        return self.np_types
    
    def read(self, source: io.RawIOBase, count=1):
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count is not int or non-positive?")
        
        dict_read = defaultdict(list)
        
        for _ in range(count):
            buf = source.read(self.size)
            
            if not buf:
                break
            
            unpacked = self._struct.unpack(buf)
            elements_zip = zip(self._names, self._sizes)
            offset = 0
            
            for name, (_, n_elems) in elements_zip:
                if isinstance(name, str):
                    dict_read[name] += unpacked[offset:(offset+n_elems)]
                offset += n_elems
        
        return {k: (v[0] if len(v) == 1 else tuple(v)) for k, v in dict_read.items()}
    
    def read_dense(self, source: io.RawIOBase, count=1) -> np.ndarray:
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count is not int or non-positive?")
        
        dtype = np.dtype(self._np_types)
        if isinstance(source, io.FileIO):
            return (np.fromfile(source, dtype=dtype, count=count))
        else:
            return np.frombuffer(
        source.read(dtype.itemsize * count), dtype=dtype, count=count
        )
            
    @staticmethod
    def _util_take_names(elements) -> tuple:
        return(tuple(name for name, *_ in elements))
    
    @staticmethod
    def _util_take_sizes(elements) -> tuple:
        def f_take():
            for _, size in elements:
                if len(size) == 1:
                    size = size + (1,)
                yield size
        
        return tuple(f_take())
    
    @staticmethod
    def _util_create_struct(cls, sizes) -> struct.Struct:
        fmts = [cls._byte_order_fmt]
        
        for type_name, count in sizes:
            count ="" if count == 1 else str(count)
            fmt, *_ = map_size_to_fmt[type_name]
            fmts.append(str(count) + fmt)
        return struct.Struct("".join(fmts))
    
    @classmethod
    def _util_create_np_types(cls, names, sizes):
        def f_name_fixer(idx, name):
            return f"__reserved{idx}__" if name is None else name
        
        bom = cls._byte_order_fmt
        types = []
        for idx, (name, (type_name, count)) in enumerate(zip(names, sizes)):
            name = f_name_fixer(idx, name)
            _, fmt, *_ = map_size_to_fmt[type_name]
            type_spec = [name, f"{bom}{fmt}"]
            if count > 1:
                type_spec += [(count,)]
            types.append(tuple(type_spec))
            
            del type_spec, fmt
            del idx, name, type_name, count
        return types
    
    @staticmethod
    def _util_gen_elements(fields, names):
        results = []
        name_index = 0
        
        for field in fields:
            part_a, part_b, *_ = field
            if part_a is None:
                results.append(tuple([None, part_b]))
            else:
                results.append(tuple([names[name_index], part_a]))
                name_index += 1
                
        return results