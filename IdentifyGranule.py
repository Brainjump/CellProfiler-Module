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
        histo_local_max, toto = np.histogram(local_max_labelize, range(object_count + 2))
        old = local_max_labelize.copy()
        print "object count = ", object_count
        print "toto = ", toto.size
        print "histo_local_max = ", histo_local_max.size
        for i in range(object_count + 1):
            value = histo_local_max[i]       
            if  value > self.range_size.max or value < self.range_size.min:
                local_max_labelize[local_max_labelize == i] = 0
        
        plt.figure()
        plt.imshow(local_max_labelize)
        plt.figure()
        plt.imshow(old)
        plt.show()
        
    def settings(self):
        return [self.primary_objects, self.file_name, self.range_size]
 
    def create_settings(self):
        self.primary_objects = cps.ObjectNameSubscriber(
            "Select the Cells object",doc="""
            Select the Cells object that you want to find the granule objects by this module.""")
        self.file_name = cps.ObjectNameProvider(
            "Name the output file to be identified",
            "Cells",doc="""
            This name will be use to create a file wich indicate the number of granule for each object.""")
        self.range_size = cps.IntegerRange(
            "Range of Granules size",(5, 40), minval=1, doc='''
            This settings permit to fix minimum and the maximumm size of the granule.''' )

#     def display(self, workspace, figure=None):
#         
#         return self