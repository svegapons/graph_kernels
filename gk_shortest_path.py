"""
Shorthest path graph kernel.

Author : Sandro Vega Pons

License: 
"""

import numpy as np
import networkx as nx
import copy
import scipy.sparse as scp
from pysparse.itsolvers import pcg
from gk_base import GK_Base
import pdb



class GK_SP(GK_Base):
    """
    Shorthest path graph kernel.
    """
    
    def compare(self, g_1, g_2, verbose=False):
        """Compute the kernel value (similarity) between two graphs. 
        
        Parameters
        ----------
        g1 : networkx.Graph
            First graph.
        g2 : networkx.Graph
            Second graph.
        alpha : interger < 1
            A rule of thumb for setting it is to take the largest power of 10
            which is samller than 1/d^2, being d the largest degree in the 
            dataset of graphs.    
            
        Returns
        -------        
        k : The similarity value between g1 and g2.
        """
        #Diagonal superior matrix of the floyd warshall shortest paths
#        pdb.set_trace()
        fwm1 = np.array(nx.floyd_warshall_numpy(g_1))
        fwm1 = np.where(fwm1==np.inf, 0, fwm1)
        fwm1 = np.where(fwm1==np.nan, 0, fwm1)
        fwm1 = np.triu(fwm1, k=1)
        bc1 = np.bincount(fwm1.reshape(-1).astype(int))
        
        fwm2 = np.array(nx.floyd_warshall_numpy(g_2))
        fwm2 = np.where(fwm2==np.inf, 0, fwm2)
        fwm2 = np.where(fwm2==np.nan, 0, fwm2)
        fwm2 = np.triu(fwm2, k=1)
        bc2 = np.bincount(fwm2.reshape(-1).astype(int))
        
        #Copy into arrays with the same length the non-zero shortests paths
        v1 = np.zeros(max(len(bc1),len(bc2)) - 1)
        v1[range(0,len(bc1)-1)] = bc1[1:]
        
        v2 = np.zeros(max(len(bc1),len(bc2)) - 1)
        v2[range(0,len(bc2)-1)] = bc2[1:]
        
        return np.sum(v1 * v2)
        
        
        
    def compare_normalized(self, g_1, g_2, verbose=False):
        """Compute the normalized kernel value between two graphs. 
        
        A normalized version of the kernel is given by the equation: 
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
        
        Parameters
        ----------
        g1 : networkx.Graph
            First graph.
        g2 : networkx.Graph
            Second graph.
        
        Returns
        -------        
        k : The similarity value between g1 and g2.
        """
        return self.compare(g_1, g_2) / (np.sqrt(self.compare(g_1, g_1) * self.compare(g_2, g_2)))

   
        
    def compare_list(self, graph_list, verbose=False):
        """Compute the all-pairs kernel values for a list of graphs. 
        
        This function can be used to directly compute the kernel matrix 
        for a list of graphs. The direct computation of the kernel matrix is
        faster than the computation of all individual pairwise kernel values.        
        
        Parameters
        ----------
        graph_list: list
            A list of graphs (list of networkx graphs)
        
        Return
        ------
        K: numpy.array, shape = (len(graph_list), len(graph_list))
        The similarity matrix of all graphs in graph_list.
        """
        n = len(graph_list)
        K = np.zeros((n,n))
        for i in range(n):
            for j in range(i, n):
                K[i,j] = self.compare(graph_list[i], graph_list[j])
                K[j,i] = K[i,j]
        return K
                
        

    def compare_list_normalized(self, graph_list, verbose=False):      
        """Compute the all-pairs kernel values for a list of graphs. 
        
        A normalized version of the kernel is given by the equation: 
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
                
        Parameters
        ----------
        graph_list: list
            A list of graphs (list of networkx graphs)
        alpha : interger < 1
            A rule of thumb for setting it is to take the largest power of 10
            which is samller than 1/d^2, being d the largest degree in the 
            dataset of graphs.     
            
        Return
        ------
        K: numpy.array, shape = (len(graph_list), len(graph_list))
        The similarity matrix of all graphs in graph_list.
        """        

        k = self.compare_list(graph_list)
        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i,j] = k[i,j] / np.sqrt(k[i,i] * k[j,j])
        
        return k_norm
        



















