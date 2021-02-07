=====================================================================================
Python Wrapper for Radar SDK
=====================================================================================

Description:
------------

This document gives some details of how to use the python wrapper (deviceControl.py)
for the radar_sdk.
This wrapper has been tested and is found to be working with Python 3.6 running on a
64-bit Windows machine. However, it is expected to work with other Python 3 versions
as well.
The file deviceControl.py mainly consists of structure and method definitions which
may be used to:..
- configure a radar device 
- connect to the device via a COM port 
- fetch raw data from the radar device in the form of frames
- disconnect from the device
File deviceControl.py essentially wraps the device control features of the radar_sdk
for use in python. Therefore it works together with the radar_sdk in the form of a 
dll.

Example:
--------

The file deviceControl.py also contains example code which illustrates how to use the
various structures and methods of the python wrapper to connect to a radar device and
how to read data from a connected radar device.

Version info:
-------------

The python wrapper is an initial version with only the most essential interfaces and
is subject to future change.

=====================================================================================
