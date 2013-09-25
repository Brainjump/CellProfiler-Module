'''
Created on 20 sept. 2013

@author: Nicolas
'''
import re
import sys
import matplotlib
import importCellProfilerLib as icpl


import scipy.ndimage
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
from cellprofiler.cpmath.otsu import otsu
import numpy as np

class MyModule(cpm.CPModule):
    """this module is a test"""
    
    #===========================================================================
    # module_name and category attributes must be added on your own classes
    #===========================================================================
    module_name = "MyModule"
    variable_revision_number = 1 #Version of the module (use for save Pipeline)
    category = "Plugin Module"
    
    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value)
        image_collection = []
        print "dimension of array = %d" % image.pixel_data.ndim
        if  not image.pixel_data.ndim == 3:
            from moduleException import ModuleException
            raise ModuleException(1, "Image must be load with the module LoadImage and the option \"individual image\"")
        image_number = image.pixel_data.shape
        print "The zvi got %d images" % image_number[2]
        for index in range(image_number[2]):
            title = ""
            if index == 0:
                title = "Original"
            elif index == 1:
                title = "Red"
            elif index == 2:
                title = "Green"
            elif index == 3:
                title = "Blue"
                 
            image_collection.append((image.pixel_data[:,:,index], title))
        
        #
        #Get the global Threshold with Otsu algorythm
        #
        global_threshold = otsu(image_collection[3][0], min_threshold=0, max_threshold=1)
        print "the threshold compute by the Otsu algorythm is %f" % global_threshold
        
        #
        #Bynary the "Blue" Image 
        #
        binary_image = np.logical_and((image_collection[3][0] >= global_threshold), image.mask)
        

        #
        #labelized the previous image.
        #
        labeled_image, object_count = scipy.ndimage.label(binary_image, np.ones((3,3), bool))
        print "the image got %d detected" % object_count
        
        new_blue = (labeled_image, image_collection[3][1])
        
        image_collection[3] = new_blue
        
        #
        #Set the image_collection attribute for display
        #
        workspace.display_data.image_collection = image_collection



    def display(self, workspace):
        """display the result on a new frame. execute after run()"""
        
        image_collection = workspace.display_data.image_collection
        
        
        if len(image_collection) == 4:
            layout = [(0,0), (1,0), (0,1), (1,1)]
        else:
            from moduleException import ModuleException
            raise ModuleException(1, "Image must be load with the module LoadImage and the option \"individual image\"")
    
        window_name = "CellProfiler:%s:%d"%(self.module_name,self.module_num)
        figure = workspace.create_or_find_figure(title="MyModule Display",
                                                 window_name = window_name,
                                                 subplots = (2,2))
        
        
        for xy, image in zip(layout, image_collection):
            figure.subplot_imshow_grayscale(xy[0], xy[1], image[0], image[1])
        
    
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