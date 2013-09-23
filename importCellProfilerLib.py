'''
Created on 19 sept. 2013

@author: FOULON Nicolas

this module permits to open the CellProfiler library, and get some module.
After import this module you can use many of the module develope by the Broad Institute team for Cell Profiler


The path, where CellProfiler is installed, must be set.
'''
import sys
import os
import _winreg

from importModuleException import ImportModuleException

is_win = sys.platform.startswith("win")

def findCellprofilerlib():
    """ find in the registre the location of Cellprofiler (for windows only)"""
    
    #TODO: add algorithm for UNIX systeme  
    if is_win:
        try:
            key = _winreg.OpenKey(_winreg.HKEY_CURRENT_USER, 'Software\\Microsoft\\IntelliPoint\\AppSpecific\\CellProfiler.exe', 0, _winreg.KEY_READ)
            (value, valuetype) = _winreg.QueryValueEx(key, 'Path')
            _winreg.CloseKey(key)
            return os.path.dirname(value)
        except WindowsError:
            raise ImportModuleException(1, "cannot find the keyregistre of CellProfiler.Are you sure Cellprofiler installed?")

    
    
def loadCellProfilerLibrary(libpath):
    """Load the library of Cell Profiler"""
    
    #test for windows system
    if is_win:
        print "\n\n", libpath, "\n\n"
        lib = "\\".join([libpath, "library.zip"])
        if not os.path.isfile(lib):
            raise ImportModuleException(2, "the file doesn't exist, check your path in \"pathlib\"file")
        sys.path.append(lib)
        
        #TODO: add test for other system


try:
    loadCellProfilerLibrary(findCellprofilerlib())
    
except ImportModuleException as e:
        sys.stderr.write(": ".join(["Error %d" % e.value, e.string]))

