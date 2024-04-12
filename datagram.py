import abc
import io
import struct
from collections import namedtuple, defaultdict

import numpy as np
import timeit


# from . import datagrams_usr

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


class DataBlock(metaclass=abc.ABCMeta):
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
        return self._struct.size
    
    @property
    def numpy_types(self):
        return self.np_types
    
    def read(self, source: io.RawIOBase, count=1):
        if not isinstance(count, int) or count <= 0:
            raise ValueError("Count is not int or non-positive?")
        
        dict_read = defaultdict(list)
        
        for _ in range(count):
            print(self.size)
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
    
    @classmethod
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


def _bytes_to_str(dict, keys):
    """
    For each key, the corresponding dict value is transformed from
    a list of bytes to a string
    """
    for key in keys:
        byte_list = dict[key]
        termination = byte_list.index(b"\x00")
        dict[key] = b"".join(byte_list[:termination]).decode("UTF-8")


def _datablock_elemblock(*items):
    """ Maps the elemBlock function on arguments before passing to Datagram """

    return DataBlock(tuple(elemBlock(*elems) for elems in items))



class Datagram(metaclass=abc.ABCMeta):
    """
    Base class for all record readers.

    Subclasses provide functionality for reading specific records.
    These are NOT the classes returned to the library user, they are only readers.
    """

    _datagram_type = None
    
    def read(self, source: io.RawIOBase):
        start_offset = source.tell()
        
        try:
            source.seek(start_offset)
            # source.seek(2, io.SEEK_CUR)
            parsed_data = self._read(source, start_offset)
            source.seek(start_offset)
            return parsed_data
        
        except AttributeError as exc:
            raise exc
        
        except ValueError as exc:
            raise exc
        
    @abc.abstractmethod
    def _read(self, source: io.RawIOBase, start_offset: int):
        raise NotImplementedError
    
    @classmethod
    def datagram_type(cls):
        """return datagram type string"""
        return cls._datagram_type
    
    
class DatagramCON0(Datagram):
    """
    Echo sounder configuration datagram
    """
    _record_type = "CON0"
    _block_dg = _datablock_elemblock(
        # DatagramHeader
        (None, binT.i32),
        ("datagram_type", binT.c8, 4),
        ("filetime", binT.u64),
        # ConfigurationHeader
        ("survey_name", binT.c8, 128),
        ("transect_name", binT.c8, 128),
        ("sounder_name", binT.c8, 128),
        ("motion_x", binT.f32),
        ("motion_y", binT.f32),
        ("motion_z", binT.f32),
        (None, binT.c8, 116),  # future use
        ("transducer_count", binT.i32)
    )
        
    _block_tdconf = _datablock_elemblock(
        # ConfigurationTransducer
        ("channel_id", binT.c8, 128),  # Channel identification
        ("beam_type", binT.i32),  # 0 = Single, 1 = Split
        ("frequency", binT.f32),  # Hz
        ("gain", binT.f32),  # dB
        ("equivalent_beam_angle", binT.f32),  # dB
        ("beamwidth_alongship", binT.f32),  # degree
        ("beamwidth_athwardship", binT.f32),  # degree
        ("angle_sensitivity_along", binT.f32),
        ("angle_sensitivity_athward", binT.f32),
        ("angle_offset_along", binT.f32),
        ("angle_offset_athward", binT.f32),
        ("pos_x", binT.f32),
        ("pos_y", binT.f32),
        ("pos_z", binT.f32),
        ("dir_x", binT.f32),
        ("dir_y", binT.f32),
        ("dir_z", binT.f32),
        (None, binT.c8, 128),
    )
    
    def _read(self, source: io.RawIOBase, start_offset: int):
        dg_header = self._block_dg.read(source)
        td_confs = []
        
        for _ in range(dg_header["transducer_count"]):
            td_confs.append(self._block_tdconf.read(source))
        
        return dg_header, td_confs

class DatagramNMEA(Datagram):
    """
    Navigation input text datagram
    """
    _record_type = "NMEA"
    _block_dg = _datablock_elemblock(
        # DatagramHeader
        ("datagram_type", binT.c8, 4),
    )
    
    def _read(self, source: io.RawIOBase, start_offset: int):
        _start_offset = source.tell()
        dg = self._block_dg.read(source)
        
        
        
        content = self._block_dg.read(source)
        return content

class DatagramTAG0(Datagram):
    """
    Annotation datagram
    """
    _record_type = "TAG0"
    _block_dg = _datablock_elemblock(
        # DatagramHeader
        ("datagram_type", binT.c8, 4),
    )
    
    def _read(self, source: io.RawIOBase, start_offset: int):
        content = self._block_dg.read(source)
        return content

class DatagramRAW0(Datagram):
    """
    Sample datagram
    """
    _record_type = "RAW0"
    _block_dg = _datablock_elemblock(
        # DatagramHeader
        ("datagram_type", binT.c8, 4),  # RAW0
        # ConfigurationHeader
        ("channel", binT.i16),  # Channel Number
        ("mode", binT.i16),  # Datatype, 
        # 0 - power sample data, 1 - power and angle sample data
        ("transducer_depth", binT.f32),  # [m]
        ("frequency", binT.f32),  # [Hz]
        ("transmit_power", binT.f32),  # [W]
        ("pulse_length", binT.f32),  # [s]
        ("bandwidth", binT.f32),  # [Hz]
        ("sample_interval", binT.f32),  # [s]
        ("sound_velocity", binT.f32),  # [m/s]
        ("absorption_coef", binT.f32),  # [dB/m]
        ("heave", binT.f32),  # [m]
        ("tx_roll", binT.f32),  # [deg]
        ("tx_pitch", binT.f32),  # [deg]
        ("temperature", binT.f32),  # [degC]
        ("spare1", binT.i16),
        ("spare2", binT.i16),
        ("rx_roll", binT.f32),  # [deg]
        ("rx_pitch", binT.f32),  # [deg]
        ("offset", binT.i32),  # First sample
        ("count", binT.i32),  # Number of samples
    )
    
    # _block_rd_amp_phs = _datablock_elemblock(
    #     ("amp", binT.i16),
    #     ("phs", binT.i16),
    # )
    
    def _read(self, source: io.RawIOBase, start_offset: int):
        content = self._block_dg.read(source)
        return content