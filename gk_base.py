"""
Base class for graph kernels.

Author : Sandro Vega Pons

License:
"""

class GK_Base(object):
    """Base class for graph kernels

    Notes
    -----
    """
    
    graphs = None
    vectors = None    

    def compare(self, g_1, g_2):
        """Compute the kernel value between the two graphs. 
        
        Parameters
        ----------
        g_1 : First graph (networkx graph or object with similar structure)
        g_2 : Second graph (networkx graph or object with similar structure)
        
        Returns
        -------        
        k : The similarity value between g_1 and g_2.
        """
        
    def compare_normalized(self, g_1, g_2):
        """Compute the normalized kernel value between two graphs. 
        
        A normalized version of the kernel is given by the equation: 
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
        
        Parameters
        ----------
        g1 : First graph (networkx graph or object with similar structure)
        g2 : Second graph (networkx graph or object with similar structure)
        
        Returns
        -------        
        k : The similarity value between g1 and g2.
        """
        
    def compare_list(self, graph_list):
        """Compute the all-pairs kernel values for a list of graphs. 
        
        This function can be used to directly compute the kernel matrix 
        for a list of graphs. For some specific graphs kernels, the direct
        computation of the kernel matrix can be faster than the computation 
        of all pairwise kernel values.        
        
        Parameters
        ----------
        graph_list: A list of graphs (list of networkx graphs)
        
        Return
        ------
        K: numpy.array, shape = (len(graph_list), len(graph_list))
        The similarity matrix of all graphs in graph_list.
        """
        
    def compare_list_normalized(self, graph_list):
        """Compute the all-pairs kernel values for a list of graphs. 
        
        A normalized version of the kernel is given by the equation: 
        k_norm(g1, g2) = k(g1, g2) / sqrt(k(g1,g1) * k(g2,g2)) 
                
        Parameters
        ----------
        graph_list: A list of graphs (list of networkx graphs)
        
        Return
        ------
        K: numpy.array, shape = (len(graph_list), len(graph_list))
        The similarity matrix of all graphs in graph_list.
        """
    
    def get_vectorial_embedding(self):
        """Gives the vectorial embedding of the previously compared graphs.
        """
        return self.vectors
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        