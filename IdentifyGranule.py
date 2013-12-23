from cellprofiler.preferences import \
    DEFAULT_INPUT_FOLDER_NAME, \
    DEFAULT_OUTPUT_FOLDER_NAME, ABSOLUTE_FOLDER_NAME
from cellprofiler.modules.loadimages import C_FILE_NAME, C_PATH_NAME
from cellprofiler.modules.identify import Identify
import cellprofiler.settings as cps

import cellprofiler
from cellprofiler.cpmath.cpmorphology import fill_labeled_holes
from cellprofiler.cpmath.otsu import otsu
import scipy.sparse
import scipy.ndimage
from skimage import filter as skf
from skimage import morphology as skm
from skimage.filter import rank as skr
import scipy.ndimage.morphology as scipym
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.morphology.watershed import watershed

#===============================================================================
# import skimage.measure as skmes
#===============================================================================


class IdentifyGranule(Identify):
    
    #===========================================================================
    # module_name and category attributes must be added on your own classes
    #===========================================================================
    module_name = "IdentifyGranule"
    variable_revision_number = 1 #Version of the module (use for save Pipeline)
    category = "Plugin Module"

    
    def run(self, workspace):
        cell_object = workspace.object_set.get_objects(self.primary_objects.value)
        cell_labeled = cell_object.get_segmented()
        cell_image = cell_object.get_parent_image()
        
        cell_image = (cell_image * 1000).astype(np.uint16)
#        object_count = cell_labeled.max()
        maxi = skr.maximum(cell_image.astype(np.uint8), skm.disk(10))
        local_max = (maxi - cell_image < 10)
        
        
        local_max_labelize, object_count = scipy.ndimage.label(local_max, np.ones((3,3), bool)) 
        histo_local_max, not_use = np.histogram(local_max_labelize, range(object_count + 2))
        old = local_max_labelize.copy()
    
        #filter in intensity mean
        #=======================================================================
        # 
        # regionprops_result = skmes.regionprops(local_max_labelize, intensity_image=cell_image)
        #      
        # for region in regionprops_result:
        #     if region["mean_intensity"]
        #=======================================================================
    
        #filter on size
        for i in range(object_count + 1):
            value = histo_local_max[i]       
            if  value > self.range_size.max or value < self.range_size.min:
                local_max_labelize[local_max_labelize == i] = 0
       
        #split granule for each cell
        cell_labeled = skm.label(cell_labeled)
        cell_count = np.max(cell_labeled)
        
        for cell_object_value in range(1, cell_count):
            cell_object_mask = (cell_labeled == cell_object_value)
            granule_in_cell = (np.logical_and(cell_object_mask, local_max_labelize))
            granule_in_cell = skm.label(granule_in_cell)
            #===================================================================
            # plt.imshow(granule_in_cell + cell_object_mask)
            # plt.show()
            #===================================================================
        #
        #get the filename
        #
        measurements = workspace.measurements
        file_name_feature = self.source_file_name_feature
        filename = measurements.get_current_measurement('Image', file_name_feature)
        print "filename = ", filename
        
        #
        #use pandalib to create the file
        #
        
    @property
    def source_file_name_feature(self):
        '''The file name measurement for the exemplar disk image'''
        return '_'.join((C_FILE_NAME, self.file_name_channel.value))
    
    def settings(self):
        return [self.primary_objects, self.file_name, self.range_size, self.location, self.file_name_channel]
 
    def create_settings(self):
        self.primary_objects = cps.ObjectNameSubscriber(
            "Select the Cells object",doc="""
            Select the Cells object that you want to find the granule objects by this module.""")
        self.file_name = cps.ObjectNameProvider(
            "Name the output file to be identified",
            "Cells",doc="""
            This name will be use to create a file wich indicate the number of granule for each object.""")
        self.range_size = cps.IntegerRange(
            "Range of Granules area (in pixel)",(5, 40), minval=1, doc='''
            This settings permit to fix minimum and the maximumm size of the granule.''' )
        
        self.location = cps.DirectoryPath(
            "Output data file location",
            dir_choices = [ABSOLUTE_FOLDER_NAME, DEFAULT_INPUT_FOLDER_NAME,
                DEFAULT_OUTPUT_FOLDER_NAME])
        self.file_name_channel = cps.ImageNameSubscriber(
            "Select the image channel to get original filename",doc = """
            get the filename of the channel from the original file.""")
        
#     def display(self, workspace, figure=None):
#         
#         return self