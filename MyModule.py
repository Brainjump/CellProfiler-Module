'''
Created on 20 sept. 2013

@author: Nicolas
'''

import matplotlib
import importCellProfilerLib

import cellprofiler.cpmodule as cpm
 
class MyModule(cpm.CPModule):
    """this module is a test"""
    module_name = "MyModule"
    category = "Plugin Module"
    def run(self, workspace):
        print "Hello i'm a new module and i'm running"
     
    def create_settings(self):
        print "Hey, i create the settings for this new module"
     
    def settings(self):
        print "Hey, i return the settings for this new module"
 
        return self.__settings
