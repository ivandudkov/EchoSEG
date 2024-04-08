from io import SEEK_CUR, BytesIO
from typing import List, Generator
from pathlib import Path

from datagram import DatagramCON0, DatagramNMEA, DatagramTAG0, DatagramRAW0
import datagrams_usr
import datetime


EPOCH_AS_FILETIME = 116444736000000000
HUNDREDS_OF_NANOSECONDS = 10000000

def ftime_to_dt(ft):
    us = (ft - EPOCH_AS_FILETIME) // 10

    # return datetime.datetime.fromtimestamp(uu)
    return datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds = us)

def btyple_to_string(bt):
    string = ''

    for bin_elem in bt:
        if bin_elem != b'\x00':
            string += bin_elem.decode()
        else:
            break
        
    return string
            


def EADatagramReader(filename: str, records_to_read: List[int] = []) -> Generator:
    """Linearly parse s7k files.
    The S7KRecordReader is a generator which linearly goes through the s7k 
    file provided in the input argument and returns one record at a time.
    For records that haven't been implemented yet, it will return
    an UnsupportedRecord, which will just include the data record frame.

    Args:
        filename (str): Name of the file to read
        records_to_read (List[int]): List of records to parse. 
            Default is the empty list representing all.

    Returns:
        A datarecord or unsupported record
    
    """

    # Ensure that the provided filepath is a string
    if not isinstance(filename, str):
        raise TypeError("Filename is not a string")

    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Filename '{filename}' could not be found!")
   
    
    with path.open(mode="rb", buffering=0) as fhandle:
        conf = DatagramCON0().read(fhandle)
        mam = datagrams_usr.Configuration(**conf)
        mam.repair_strings()
        print(mam)
        # print(btyple_to_string(mam.sounder_name))
        print(f"time is {mam.filetime_py()}")
        # while True:
        #     # Otherwise read the record data and the next data record frame.
        #     # This way we can handle the read linearly
        #     raw_bytes = fhandle.read(drf.size)

        #     # Because the DataRecord read functionality assumes that we are 
        #     # reading from the start of the record, we'll prepend the raw
        #     # bytes with the size of the data record frame
        #     record_content_bytes = BytesIO(DRF_START_SIZED_DUMMY + raw_bytes)
        #     record = _datarecord.record(drf.record_type_id).read(record_content_bytes, drf)
            
        #     if len(raw_bytes) < drf.size:
        #         # If the size of the data record frame is greater than the size
        #         # of the raw bytes it means that there is no more data, indicating
        #         # EOF
        #         drf = None
        #     else:
        #         # Read the next data record frame
        #         drf = DRFBlock().read(BytesIO(raw_bytes[-DRF_BYTE_SIZE:]))
        #     yield record