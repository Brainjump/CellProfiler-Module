'''
Created on Nov 15, 2013

@author: nicolas
'''
import re
import sys
import matplotlib
#import importCellProfilerLib as icpl

import scipy.sparse
import scipy.ndimage
from cellprofiler.modules.identify import Identify
import cellprofiler.settings as cps
from cellprofiler.cpmath.cpmorphology import fill_labeled_holes
from cellprofiler.cpmath.otsu import otsu
from skimage import filter as skf
from skimage import morphology as skm
from skimage.filter import rank as skr
import scipy.ndimage.morphology as scipym
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import cellprofiler
from skimage.morphology.watershed import watershed

class IdentifyCellsWithNuclei(Identify):
    """this module is a test"""
    
    #===========================================================================
    # module_name and category attributes must be added on your own classes
    #===========================================================================
    module_name = "IdentifyCellWithNuclei"
    variable_revision_number = 1 #Version of the module (use for save Pipeline)
    category = "Plugin Module"

    
    def run(self, workspace):
        labeled_nuclei = workspace.object_set.get_objects(self.primary_objects.value).get_segmented()
        cell_image = workspace.image_set.get_image(self.image_name.value).pixel_data[:,:]
        image_collection = []
        cell_treshold = otsu(cell_image, min_threshold=0, max_threshold=1)
        
        cell_binary = (cell_image >= cell_treshold)
        cell_distance = scipym.distance_transform_edt(cell_binary).astype(np.uint16)
        cell_labeled = skm.watershed(-cell_distance, labeled_nuclei, mask=cell_binary)
        
         
        #
        #fil hall and filter on syze the object in cell_labeled
        #
        cell_labeled = self.filter_on_border(cell_labeled)
        cell_labeled = fill_labeled_holes(cell_labeled)
    
        objects = cellprofiler.objects.Objects()
        objects.segmented = cell_labeled
        objects.parent_image = cell_image
        
        workspace.object_set.add_objects(objects, self.object_name.value)        
        image_collection.append((cell_image, "Original"))
        image_collection.append((cell_labeled, "Labelized image"))
        workspace.display_data.image_collection = image_collection
        
    def display(self, workspace, figure=None):
        """display the result on a new frame. execute after run()"""
        
        image_collection = workspace.display_data.image_collection
        if len(image_collection) == 2:
            layout = [(0,0),  (0,1)]
        else:
            from moduleException import ModuleException
            raise ModuleException(1, "Image must be load with the module LoadImage and the option \"individual image\"")
        
        if figure is not None:
            figure.set_subplots((1,2))
        else:
            window_name = "CellProfiler:%s:%d"%(self.module_name,self.module_num)
            figure = workspace.create_or_find_figure(title="MyModule Display",
                                                     window_name = window_name,
                                                     subplots = (2,2))
        
        for xy, image in zip(layout, image_collection):
            if xy == (0,1):
                figure.subplot_imshow_labels(xy[0], xy[1], image[0], image[1])
            else:
                figure.subplot_imshow_grayscale(xy[0], xy[1], image[0], image[1])
        
    def create_settings(self):
        """This method create the settings. This use to set the diverse buttons"""
        
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image",doc = """
            Choose the image to be cropped.""")
        self.primary_objects = cps.ObjectNameSubscriber(
            "Select the input object",doc="""
            Select the nuclei object that you want to find the cells objects by this module.""")
        self.object_name = cps.ObjectNameProvider(
            "Name the Cell objects to be identified",
            "Cells",doc="""
            Enter the name that you want to call the objects identified by this module.""")
        
    def settings(self):
        """This method return the settings set"""
        return [self.image_name, self.primary_objects, self.object_name]
    def filter_on_border(self, labeled_image):
        #
        #Get outline's pixel of the labeled_image
        #
        border_labels = list(labeled_image[0,:])
        border_labels.extend(labeled_image[:,0])
        border_labels.extend(labeled_image[labeled_image.shape[0]-1,:])
        border_labels.extend(labeled_image[:,labeled_image.shape[1]-1])
        border_labels = np.array(border_labels)
        
        #
        #Get the histogram
        #First: create a coo_matrix (it's just a format: 3 columns. 1 for the row, 1 for the column, the last for the value)
        #here row[i] = border_label[i], column[i] = 0, value[i] = 1
        # todense() create the matrix. create a Matrix initialized at 0, with the shape given.
        #then add the value[i] at the position: (row[i], column[i])
        #the trick here is column value is always 0, and the value always 1.
        #So that create the histogram. flatten just convert the matrix into array.
        #the length of the histogram = number_of_object + 1
        #
        
        histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape),
                                             (border_labels,
                                              np.zeros(border_labels.shape))),
                                              shape=(np.max(labeled_image)+1,1)).todense()
                                             
        histogram = np.array(histogram).flatten()
        
        if np.any(histogram[1:] > 0 ):
            histogram_image = histogram[labeled_image]
            labeled_image[histogram_image > 0] = 0
        return labeled_image
    
    def filter_on_size(self, labeled_image, object_count):
        areas = scipy.ndimage.measurements.sum(np.ones(labeled_image.shape),
                                               labeled_image,
                                               range(0, object_count+1))
        areas = np.array(areas, dtype=int)
        
        #set min_size to 40
        min_allowed_area = np.pi * (40 * 40) / 4
        
        area_image = areas[labeled_image]
        labeled_image[area_image < min_allowed_area] = 0
        
        return labeled_image


    def calc_smoothing_filter_size(self):
        #must be change if setting ask for min size of object
        range_min = 40
        return 2.35 * range_min / 3.5