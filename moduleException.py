'''
Created on 24 sept. 2013

@author: Nicolas
'''

class ModuleException(Exception):
    """Exception class for module
        
        This Module contain:
        """
    def __init__(self, value, string=""):
        self.value = value
        self.string = string
    
    def __str__(self):
        return self.string
    
    def printException(self):
        print "Error %d: %s" % self.value, self.string