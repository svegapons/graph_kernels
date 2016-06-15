"""
Base class for graph kernels.

Author : Sandro Vega Pons
"""

import numpy as np


class GK_Base(object):
    """Base class for graph kernels
    """  
        
    def compare_pairwise(self, graph_list, normalize=True):
        """Compute the all-pairs kernel values for a list of graphs. 
        
        Parameters
        ----------
        graph_list: list of ndarray
                    A list of graphs (adjacency matrices)
        
        Return
        ------
        K: ndarray, shape = (len(graph_list), len(graph_list))
            The similarity matrix of all graphs in graph_list.
        """
        #To be defined by each graph kernel.

        
        
    def _normalize(self, K):
        """Normalize kernel matrix K by using:         
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
                
        Parameters
        ----------
        K: ndarray
           Kernel matrix to be normalized
        
        Return
        ------
        K_norm: ndarray
            Normalized kernel matrix
        """
        K_norm = np.zeros(K.shape)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                K_norm[i,j] = K[i,j] / np.sqrt(K[i,i] * K[j,j])        
        return K_norm
    

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        