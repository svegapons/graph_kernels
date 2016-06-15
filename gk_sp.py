"""
Shorthest path graph kernel.

Author : Sandro Vega Pons
"""

import numpy as np
import networkx as nx
from gk_base import GK_Base


class GK_SP(GK_Base):
    """
    Shorthest path graph kernel.
    """
    
    def __init__(self, th=0., binary_edges=True):
        """
        Parameters
        ----------
        th: float
            Threshold to edge weights
        binary_edges: boolean
            Wheather to use binay edges (after thresholding with th) or keep
            the original edge weights (the ones that survived the thresholding)
        """
        self.th = th
        self.binary_edges = binary_edges
        
    
    def _compare(self, g1, g2):
        """Compute the kernel value between the two graphs. 
        
        Parameters
        ----------
        g1 : ndarray
            Adjacency matrix of the first graph.
        g2 : ndarray
            Adjacency matrix of the second graph.
            
        Returns
        -------        
        k : The similarity value between g1 and g2.
        """
        if self.binary_edges:
            g1 = np.where(g1 > self.th, 1, 0)
            g2 = np.where(g2 > self.th, 1, 0)
        else:
            g1 = np.where(g1 > self.th, g1, 0)
            g2 = np.where(g2 > self.th, g2, 0)
        
        g1 = nx.from_numpy_matrix(g1)
        g2 = nx.from_numpy_matrix(g2)
        
        #Diagonal superior matrix of the floyd warshall shortest paths
        fwm1 = np.array(nx.floyd_warshall_numpy(g1))
        fwm1 = np.where(fwm1==np.inf, 0, fwm1)
        fwm1 = np.where(fwm1==np.nan, 0, fwm1)
        fwm1 = np.triu(fwm1, k=1)
        bc1 = np.bincount(fwm1.reshape(-1).astype(int))
        
        fwm2 = np.array(nx.floyd_warshall_numpy(g2))
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
        n = len(graph_list)
        K = np.zeros((n,n))
        for i in range(n):
            for j in range(i, n):
                K[i,j] = self._compare(graph_list[i], graph_list[j])
                K[j,i] = K[i,j]
                
        if normalize:
            K = self._normalize(K)
            
        return K















