from dataclasses import dataclass, field
import datetime
from enum import Enum
from typing import Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np

def btyple_to_string(bt):
    string = ''

    for bin_elem in bt:
        if bin_elem != b'\x00':
            string += bin_elem.decode()
        else:
            break
        
    return string


EPOCH_AS_FILETIME = 116444736000000000
HUNDREDS_OF_NANOSECONDS = 10000000

def ftime_to_dt(ft):
    us = (ft - EPOCH_AS_FILETIME) // 10

    # return datetime.datetime.fromtimestamp(uu)
    return datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds = us)


@dataclass
class Configuration:
    """
    somesomesome
    """
    
    datagram_type: int
    filetime: int
    survey_name: str
    transect_name: str
    sounder_name: str
    motion_x: float
    motion_y: float
    motion_z: float
    
    transducer_count: int
    channel_id: int
    beam_type: int
    frequency: int
    gain: float
    equivalent_beam_angle: float
    beamwidth_alongship: float
    beamwidth_athwardship: float
    angle_sensitivity_along: float
    angle_sensitivity_athward: float
    angle_offset_along: float
    angle_offset_athward: float
    pos_x: float
    pos_y: float
    pos_z: float
    dir_x: float
    dir_y: float
    dir_z: float



    def repair_strings(self):
        self.datagram_type = btyple_to_string(self.datagram_type)
        self.survey_name = btyple_to_string(self.survey_name)
        self.transect_name = btyple_to_string(self.transect_name)
        self.sounder_name = btyple_to_string(self.sounder_name)
        self.channel_id = btyple_to_string(self.channel_id)
        
        
    def filetime_py(self):
        return(ftime_to_dt(self.filetime))


