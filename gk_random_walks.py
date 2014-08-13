"""
Random walks graph kernel.

Python implementation of Nino Shervashidze Matlab code at:
http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/

Author : Sandro Vega Pons

License: 
"""

import numpy as np
import networkx as nx
import copy
from pysparse.itsolvers import pcg
from gk_base import GK_Base



class GK_RW(GK_Base):
    """
    Random walks graph kernel.
    """
    
    def compare(self, g1, g2, alpha, verbose=False):
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
        am1 = nx.adj_matrix(g1)
        am2 = nx.adj_matrix(g2)
        x = np.zeros((len(am1),len(am2)))
        A = self.smt_filter(x,am1,am2,alpha)
        b = np.ones(len(am1)*len(am2))
        tol = 1e-6
        maxit = 20
        pcg(A,b,x,tol,maxit)
        return np.sum(x)
        
        
    def smt_filter(self, x, am1, am2, alpha):
        yy = np.dot(np.dot(am1, x), am2)
        yy = yy * alpha
        vecu = x - yy
        return vecu

        
        
    def compare_normalized(self, g1, g2, alpha, verbose=False):
        """Compute the normalized kernel value between two graphs. 
        
        A normalized version of the kernel is given by the equation: 
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
        
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

       
        
    def compare_list(self, graph_list, alpha, verbose=False):
        """Compute the all-pairs kernel values for a list of graphs. 
        
        This function can be used to directly compute the kernel matrix 
        for a list of graphs. The direct computation of the kernel matrix is
        faster than the computation of all individual pairwise kernel values.        
        
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
        
        n = len(graph_list)
        K = np.zeros((n,n))
        for i in range(n-1):
            for j in range(i+1, n):
                K[i,j] = self.compare(graph_list[i], graph_list[j], alpha)
                K[j,i] = K[i,j]
        return K
                
        

    def compare_list_normalized(self, graph_list, alpha, verbose=False):      
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

        k = self.compare_list(graph_list, alpha)
        k_norm = np.zeros(k.shape)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                k_norm[i,j] = k[i,j] / np.sqrt(k[i,i] * k[j,j])
        
        return k_norm
        

