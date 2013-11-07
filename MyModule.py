'''
Created on 20 sept. 2013

@author: Nicolas
'''
import re
import sys
import matplotlib
#import importCellProfilerLib as icpl

import scipy.sparse
import scipy.ndimage
import cellprofiler.cpmodule as cpm
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
        #
        #Test if the image was open with option "individual image" in module LoadImages
        #
        if  not image.pixel_data.ndim == 3:
            from moduleException import ModuleException
            raise ModuleException(1, "Image must be load with the module LoadImage and the option \"individual image\"")
        image_number = image.pixel_data.shape
        print "The zvi got %d images" % image_number[2]
        
        #
        #Creation of the image_collection
        #
        for index in range(image_number[2]):
            title = ""
            if index == 0:
                title = "Original"
            elif index == 1:
                title = "GFP-M9M"
            elif index == 2:
                title = "elF38"
            elif index == 3:
                title = "Dapi"
                 
            image_collection.append((image.pixel_data[:,:,index], title))
        
        
        image_collection[1] = (image_collection[3][0], "original Elf3B")
        image_collection[0] = (image_collection[2][0], "original DAPI")
 
        #
        #Get the global Threshold with Otsu algorithm and smooth nuclei image
        #
#         nuclei_smoothed = self.smooth_image(image_collection[3][0], image.mask, 1)
        nuclei_smoothed = image_collection[3][0] 
        global_threshold_nuclei = otsu(nuclei_smoothed, min_threshold=0, max_threshold=1)
        print "the threshold compute by the Otsu algorythm is %f" % global_threshold_nuclei      
        
        
        #
        #Binary the "DAPI" Image (Nuclei) and labelelize the nuclei
        #

        binary_nuclei = (nuclei_smoothed >= global_threshold_nuclei)
        labeled_nuclei, object_count = scipy.ndimage.label(binary_nuclei, np.ones((3,3), bool))
        print "the image got %d detected" % object_count
        
        #
        #Fill the hole and delete object witch touch the border. 
        #labeled_nuclei is modify after the function
        #Filter small object and split object
        #
        labeled_nuclei = fill_labeled_holes(labeled_nuclei)        
        labeled_nuclei = self.filter_on_border(labeled_nuclei)
        labeled_nuclei = self.filter_on_size(labeled_nuclei, object_count)         
        labeled_nuclei = self.split_object(labeled_nuclei)
        
        #
        #Edge detection of nuclei image and object are more separated 
        #
        labeled_nuclei_canny = skf.sobel(labeled_nuclei)       
        labeled_nuclei[labeled_nuclei_canny > 0] = 0
        labeled_nuclei = skr.minimum(labeled_nuclei.astype(np.uint16), skm.disk(3))
        
        # 
        #segmentation of the Cell in the Elf3B image
        #      
        cell_image = image_collection[0][0]
        
        cell_threshold = otsu(cell_image,
                              min_threshold=0,
                              max_threshold=1)
        cell_bg_mark = skm.binary_erosion((cell_image < cell_threshold).astype(np.uint8),                                   
                                          skm.disk(5))
        

        marker = labeled_nuclei.copy()
        number_labeled_nuclei = np.max(marker)
        marker[cell_bg_mark] = number_labeled_nuclei + 1
        
        cell_gradient = skr.gradient(cell_image.astype(np.uint8), skm.disk(3))
        
        print "cell threshold = ", cell_threshold


#         cell_binary = skr.maximum(cell_binary.astype(np.uint16), skm.disk(5))
#         cell_binary = fill_labeled_holes(cell_binary)
#         cell_binary = skm.binary_erosion(cell_binary, skm.disk(5))
#         distance = scipym.distance_transform_edt(cell_binary).astype(np.uint16)
#         labeled_nuclei[labeled_nuclei == labeled_nuclei[0][0]] = 0
        
        #Set the image_collection attribute for display
        
        
        labeled_cell = skm.watershed(cell_gradient, marker)
         
        new_labeled_elf3b = (labeled_cell, image_collection[2][1])

        new_labeled_dapi = (labeled_nuclei, image_collection[3][1])
        
        canny_for_test = image_collection[1][0].copy()
        canny_for_test = skf.sobel(canny_for_test)
#         canny_for_test_otsu = otsu(canny_for_test, min_threshold=0, max_threshold=1)
#         canny_for_test = (canny_for_test >= canny_for_test_otsu)
        
        plt.figure()
        plt.imshow(canny_for_test)
        plt.show()
        
        image_collection[2] = new_labeled_elf3b
        image_collection[3] = new_labeled_dapi
        workspace.display_data.image_collection = image_collection


    def display(self, workspace, figure=None):
        """display the result on a new frame. execute after run()"""
        
        image_collection = workspace.display_data.image_collection
        if len(image_collection) == 4:
            layout = [(0,0), (1,0), (0,1), (1,1)]
        else:
            from moduleException import ModuleException
            raise ModuleException(1, "Image must be load with the module LoadImage and the option \"individual image\"")
        
        if figure is not None:
            figure.set_subplots((2,2))
        else:
            window_name = "CellProfiler:%s:%d"%(self.module_name,self.module_num)
            figure = workspace.create_or_find_figure(title="MyModule Display",
                                                     window_name = window_name,
                                                     subplots = (2,2))
        
        
        for xy, image in zip(layout, image_collection):
            if xy == (1,1) or xy == (0,1):
                figure.subplot_imshow_labels(xy[0], xy[1], image[0], image[1])
            else:
                figure.subplot_imshow_grayscale(xy[0], xy[1], image[0], image[1])
        #=======================================================================
        # for thread in self.thread_list:
        #     thread.wait()
        #=======================================================================
    
    def create_settings(self):
        """This method create the settings. This use to set the diverse buttons"""
        
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image",doc = """
            Choose the image to be cropped.""")

     
    def settings(self):
        """This method return the settings set"""
        #print "Hey, i return the settings for this new module"
        return [self.image_name]
 
        return self.__settings

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
        # todense() create the matrixtwitter rebublic of gamer. create a Matrix initialized at 0, with the shape given.
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
        
    def plot_image(self, image_list):
        for image in image_list:
            plt.figure()
            plt.imshow(image)
            plt.show()
            
    def split_object(self, labeled_image):
        """ split object when it's necessary
        """
        
        labeled_image = labeled_image.astype(np.uint16)

        labeled_mask = np.zeros_like(labeled_image, dtype=np.uint16)
        labeled_mask[labeled_image != 0] = 1

        #ift structuring element about center point. This only affects eccentric structuring elements (i.e. selem with even num===============================
       
        labeled_image = skr.median(labeled_image, skm.disk(4))
        labeled_mask = np.zeros_like(labeled_image, dtype=np.uint16)
        labeled_mask[labeled_image != 0] = 1
        distance = scipym.distance_transform_edt(labeled_image).astype(np.uint16)
        #=======================================================================
        # binary = np.zeros(np.shape(labeled_image))
        # binary[labeled_image > 0] = 1
        #=======================================================================
        distance = skr.mean(distance, skm.disk(15))
         
        l_max = skr.maximum(distance, skm.disk(5))
        #l_max = skf.peak_local_max(distance, indices=False,labels=labeled_image, footprint=np.ones((3,3)))
        l_max = l_max - distance <= 0
        
        l_max = skr.maximum(l_max.astype(np.uint8), skm.disk(6))
        
       
        
        marker = ndimage.label(l_max)[0]
        split_image = skm.watershed(-distance, marker)
        
        return split_image
        
    def smooth_image(self, image, mask, sigma):
        
        #
        #Code from CellProfiler
        #Need explanantion
        #
        
        filter_size = self.calc_smoothing_filter_size() 
        
        filter_size = max(int(float(filter_size) / 2.0),1)
        f = (1/np.sqrt(2.0 * np.pi ) / sigma *
             np.exp(-0.5 * np.arange(-filter_size, filter_size+1)**2 /
                    sigma ** 2))
        def fgaussian(image):
            output = scipy.ndimage.convolve1d(image, f,
                                              axis = 0,
                                              mode='constant')
            return scipy.ndimage.convolve1d(output, f,
                                            axis = 1,
                                            mode='constant')
        #
        # Use the trick where you similarly convolve an array of ones to find
        # out the edge effects, then divide to correct the edge effects
        #
        edge_array = fgaussian(mask.astype(float))
        masked_image = image.copy()
        masked_image[~mask] = 0
        smoothed_image = fgaussian(masked_image)
        masked_image[mask] = smoothed_image[mask] / edge_array[mask]
        return masked_image
    
    def calc_smoothing_filter_size(self):
        #must be change if setting ask for min size of object
        range_min = 40
        return 2.35 * range_min / 3.5
    