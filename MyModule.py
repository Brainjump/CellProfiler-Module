'''
Created on 20 sept. 2013

@author: Nicolas
'''
import re
import sys
import matplotlib
import importCellProfilerLib as icpl

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

    
class MyModule(cpm.CPModule):
    """this module is a test"""
    
    #===========================================================================
    # module_name and category attributes must be added on your own classes
    #===========================================================================
    module_name = "MyModule"
    category = "Plugin Module"
    def run(self, workspace):
        print "Hello i'm a new module and i'm running"
        #=======================================================================
        # image = workspace.image_set.get_image(self.image_name.value,
        #                                       must_be_grayscale = True)
        #=======================================================================
        window_name = "CellProfiler:%s:%d"%(self.module_name,self.module_num)
        figure = workspace.create_or_find_figure(title="MyModule Display",
                                                 window_name = window_name,
                                                 subplots = (1,1))
        image = workspace.image_set.get_image(self.image_name.value)
        title = "Dysplay"
        figure.subplot_imshow_grayscale(0,0, image.pixel_data, title)
        
        
    def create_settings(self):
        """This method create the settings. This use to set the diverse buttons"""
        
        #print "Hey, i create the settings for this new module"
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image",doc = """
            Choose the image to be cropped.""")
     
    def settings(self):
        """This method return the settings set"""
        #print "Hey, i return the settings for this new module"
        return [self.image_name]
 
        return self.__settings

    #===========================================================================
    # def display(self, workspace):
    #     figure = workspace.create_or_find_figure(title="MyModule Display", )
    #===========================================================================