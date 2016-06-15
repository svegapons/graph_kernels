"""
Weisfeiler_Lehman graph kernel.

Python implementation of Nino Shervashidze Matlab code at:
http://mlcb.is.tuebingen.mpg.de/Mitarbeiter/Nino/Graphkernels/

Author : Sandro Vega Pons

"""

import numpy as np
import networkx as nx
import copy
from gk_base import GK_Base
import pdb


class GK_WL(GK_Base):
    """
    Weisfeiler_Lehman graph kernel.
    """
    
    def __init__(self, th=0, h=1):
        """
        Parameters
        ----------
        th: float
            Threshold to edge weights
        h : int
            Number of iterations
        nl : boolean
            Whether to use original node labels. True for using node labels
            saved in the attribute 'node_label'. False for using the node 
            degree of each node as node attribute.
        """
        self.th = th
        self.h = h       
       
        
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
        #Thresholding and computing the networkx graphs
        graph_list = [nx.from_numpy_matrix(np.where(mat > self.th, 1, 0)) 
                      for mat in graph_list]
        n = len(graph_list)
        lists = [0] * n
        k = [0]*(self.h + 1)
        n_nodes = 0
        n_max = 0
        
        #Compute adjacency lists and n_nodes, the total number of nodes in
        #the dataset.
        for i in range(n):
            #the graph should be a networkx graph or having the same methods.
            lists[i] = graph_list[i].adjacency_list()
            n_nodes = n_nodes + graph_list[i].number_of_nodes()
            
            #Computing the maximum number of nodes in the graphs. It will be 
            #used in the computation of vectorial representation.
            if(n_max < graph_list[i].number_of_nodes()):
                n_max = graph_list[i].number_of_nodes()
            
        phi = np.zeros((n_max, n))
        #each column j of phi will be the explicit feature representation
        # for the graph j.
        #n_max is enough to store all possible labels
        
        #INITIALIZATION
        #initialize the nodes labels for each graph with their labels or 
        #with degrees (for unlabeled graphs)
        
        labels = [0] * n
        label_lookup = {}
        label_counter = 0

        # label_lookup is an associative array, which will contain the
        # mapping from multiset labels (strings) to short labels (integers)

        #Node degrees are used as node labels
        for i in range(n):
            labels[i] = np.array(graph_list[i].degree().values())    
            for j in range(len(labels[i])):
                phi[labels[i][j], i] += 1
        
        k = np.dot(phi.transpose(), phi)
        
        
        ### MAIN LOOP
        it = 0
        new_labels = copy.deepcopy(labels)
        
        
        while it < self.h:
            # create an empty lookup table
            label_lookup = {}
            label_counter = 0

            phi = np.zeros((n_nodes, n))
            for i in range(n):
                for v in range(len(lists[i])):
                    # form a multiset label of the node v of the i'th graph
                    # and convert it to a string
#                    pdb.set_trace()
                    long_label = np.concatenate((np.array([labels[i][v]]), 
                                                 np.sort(labels[i]
                                                 [lists[i][v]])))
                    long_label_string = str(long_label)
                    # if the multiset label has not yet occurred, add it to the
                    # lookup table and assign a number to it
                    if not label_lookup.has_key(long_label_string):
                        label_lookup[long_label_string] = label_counter
                        new_labels[i][v] = label_counter
                        label_counter += 1
                    else:
                        new_labels[i][v] = label_lookup[long_label_string]
                # fill the column for i'th graph in phi
                aux = np.bincount(new_labels[i])
                phi[new_labels[i], i] += aux[new_labels[i]]
            
            k += np.dot(phi.transpose(), phi)
            labels = copy.deepcopy(new_labels)
            it = it + 1
            
        if normalize:
            k = self._normalize(k)            
            
        return k
        
                    