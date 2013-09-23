'''
Created on 19 sept. 2013

@author: Nicolas

Exception class for importCellProfilerLib
'''


class ImportModuleException(Exception):
    """ Exception class for importCellProfilerLib"""

    def __init__(self, value, string=""):
        self.value = value
        self.string = string
        
    def __str__(self):
        return repr(self.value)
