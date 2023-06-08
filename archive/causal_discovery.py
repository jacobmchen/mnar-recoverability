"""
This file is adapted from Homework 4 of CSCI 379 Causal Inference of Fall 2021 taught by
Rohit Bhattacharya.
"""

from venv import create
import numpy as np
import pandas as pd
from collections import defaultdict
import statsmodels.api as sm
from statsmodels.formula.api import ols
import random
import itertools
from shadow_ygraph_fixing import testShadowYGraph

def findSubsets(s, n):
    """
    s is a set of vertices
    n is the size of the subset
    """
    return list(itertools.combinations(s, n))

class Graph():
    """
    A class for creating graph objects. A graph will be
    defined by (i) a set of vertices and (ii) a dictionary
    mapping each vertex Vi to its set of parents pa_G(Vi).
    """

    def __init__(self, vertices, edges=set()):
        """
        Constructor for the Graph class. A Graph is created
        by accepting a set of vertices and edges as inputs

        Inputs:
        vertices: a set/list of vertex names
        edges: a set/list of tuples for example [(Vi, Vj), (Vj, Vk)] where the tuple (Vi, Vj) indicates Vi->Vj exists in G
        """
        self.vertices = set(vertices)
        self.parents = defaultdict(set)
        for parent, child in edges:
            self.parents[child].add(parent)

    def add_edge(self, parent, child):
        """
        Function to add an edge to the graph from parent -> child
        """

        self.parents[child].add(parent)

    def delete_edge(self, parent, child):
        """
        Function to delete an edge to the graph from parent -> child
        """

        if parent in self.parents[child]:
            self.parents[child].remove(parent)

    def edges(self):
        """
        Returns a list of tuples [(Vi, Vj), (Vx, Vy),...] corresponding to edges
        present in the graph
        """

        edges = []
        for v in self.vertices:
            edges.extend([(p, v) for p in self.parents[v]])

        return edges

    def vertexList(self):
        """
        Returns a list of vertices corresponding to vertices present in the graph
        """
        vertexList = []
        for v in self.vertices:
            vertexList.append(v)

        return vertexList

    def parentList(self, vertex):
        """
        Returns the list of parents for vertex.
        """
        parentList = []
        for v in self.parents[vertex]:
            parentList.append(v)

        return parentList

    def produce_visualization_code(self, filename):
        """
        Function that outputs a text file with the necessary graphviz
        code that can be pasted into https://dreampuf.github.io/GraphvizOnline/
        to visualize the graph.
        """

        # set up a Digraph object in graphviz
        gviz_file = open(filename, "w")
        gviz_file.write("Digraph G { \n")

        self.vertices = sorted(self.vertices)

        # iterate over all vertices and add them to the graph
        for v in self.vertices:
            gviz_file.write('  {} [shape="plaintext"];\n'.format(v))

        # add edges between the vertices
        for v in self.vertices:
            for p in self.parents[v]:
                gviz_file.write('  {} -> {} [color="blue"];\n'.format(p, v))

        # close the object definition and close the file
        gviz_file.write("}\n")
        gviz_file.close()

def acyclic(G):
    """
    A function that uses depth first traversal to determine whether the
    graph G is acyclic.
    """
    
    # If G_transpose has a cycle, then G will also have a cycle.
    # Traverse G_transpose by going to each of a vertex's parents.

    # Start from every vertex and use a DFS to see if we can return to
    # the original vertex.
    vertexList = G.vertexList()

    for startVertex in vertexList:
        # Do a DFS from startVertex.

        # visited keeps track of which vertices have been visited and have
        # not been visited yet.
        visited = {}
        for v in vertexList:
            visited[v] = 0
        
        stk = []
        stk.append(startVertex)
        while len(stk) > 0:
            curVertex = stk.pop()
            # Visit all of the parents of curVertex.
            for v in G.parentList(curVertex):
                if v == startVertex:
                    # There is a cycle!
                    return False
                if visited[v] == 0:
                    stk.append(v)
            # Mark curVertex as visited
            visited[curVertex] = 1

    # We did not find a cycle starting from any vertex.
    return True

def bic_score(G, data):
    """
    Compute the BIC score for a given graph G and a dataset as a pandas data frame.

    Inputs:
    G: a Graph object as defined by the Graph class above
    data: a pandas data frame
    """

    # Calculate a BIC score for each vertex v by fitting it as a function of
    # an intercept and its parents. Sum all of the BIC scores to obtain a
    # BIC score for the entire graph.
    bicsum = 0
    vertexList = G.vertexList()

    for vertex in vertexList:
        parents = G.parentList(vertex)
        # If the vertex has no parents, fit it with only an intercept.
        if len(parents) == 0:
            formula = vertex + "~1"
        else:
            # Fit the vertex as a function of its parents.
            formula = vertex + "~"
            for parent in parents:
                formula += "+" + parent

        model = ols(formula=formula, data=data).fit()
        # Add the BIC score to the sum.
        bicsum += model.bic

    return bicsum

def causal_discovery(data, must_edges=[], num_steps=50, verbose=False):
    """
    Take in data and perform causal discovery according to a set of moves
    described in the write up for a given number of steps.

    must_edges is a list of edges that are not allowed to be deleted and must
    remain in the graph
    """

    # initalize a graph with the required edges as the optimal one and 
    # gets its BIC score
    G_star = Graph(vertices=data.columns, edges=must_edges)
    bic_star = bic_score(G_star, data)
    vertices = G_star.vertexList()
    vertices = sorted(vertices)
    numVertices = len(vertices)

    # forward phase of causal discovery:
    for i in range(num_steps):
        if verbose:
            print("iteration", i, G_star.edges())

        # attempt a random edge addition that does not create a cycle
        # and we repeat this procedure until we find an edge that works.
        edges_starList = G_star.edges()
        G_add = Graph(vertices, edges_starList)
        while True:
            # pick two random vertices v_i and v_j
            v1 = random.randint(0, numVertices-1)
            v2 = random.randint(0, numVertices-1)

            # test if the edge already exists
            if (vertices[v1], vertices[v2]) not in edges_starList:
                G_add.add_edge(vertices[v1], vertices[v2])
                if acyclic(G_add):
                    # This is a valid edge to add.
                    break
                else:
                    # This is not a valid edge to add. Delete it and search for another edge.
                    G_add.delete_edge(vertices[v1], vertices[v2])

        # if it improves the BIC score, update G_star and bic_star
        # calculate the BIC score for G_add
        bic_add = bic_score(G_add, data)
        if bic_add < bic_star:
            # G_add is a better DAG, so update G_star
            G_star = G_add
            bic_star = bic_add


    # backward phase of causal discovery
    for i in range(num_steps):
        if verbose:
            print("iteration", i, G_star.edges())

        # attempt a random edge deletion/reversal
        # choose a random edge in G_star
        edges_starList = G_star.edges()
        edgeIndex = random.randint(0, len(edges_starList)-1)
        chosenEdge = edges_starList[edgeIndex]

        # check that chosenEdge is not in mustEdge, if it is then skip
        # this iteration
        while chosenEdge in must_edges:
            edgeIndex = random.randint(0, len(edges_starList)-1)
            chosenEdge = edges_starList[edgeIndex]
        if verbose:
            print(chosenEdge)

        # G_del deletes the chosen edge from the graph
        G_del = Graph(vertices, edges_starList)
        G_del.delete_edge(chosenEdge[0], chosenEdge[1])

        # G_rev reverses the chosen edge in the graph
        G_rev = Graph(vertices, edges_starList)
        G_rev.delete_edge(chosenEdge[0], chosenEdge[1])
        G_rev.add_edge(chosenEdge[1], chosenEdge[0])
        # have to check that G_rev doesn't create any cycles
        G_revIsAcyclic = acyclic(G_rev)

        # pick the move that improves the BIC score (if any)
        bic_del = bic_score(G_del, data)
        bic_rev = bic_score(G_rev, data)
        if verbose:
            print(bic_star, bic_del, bic_rev)
            print()
        if G_revIsAcyclic and bic_rev <= bic_star and bic_rev <= bic_del:
            # reversing the edge has the lowest bic score
            G_star = G_rev 
            bic_star = bic_rev 
        elif bic_del < bic_star:
            # deleting the edge has the lowest bic score
            G_star = G_del 
            bic_star = bic_del
        # else both deleting and reversing don't improve the bic score

    return G_star

def generateEdges(vertices):
    """
    Generate all possible edges from a graph with vertices vertices. If n is the 
    total number of vertices, there are n choose 2 possible edges.
    vertices is a list of strings representing the vertices.
    Returns the vertices as a list of tuples where the tuples are in the form of
    (parent, child).
    """
    n = len(vertices)
    edges = []

    for i in range(n):
        for j in range(i, n):
            if i != j:
                edges.append((vertices[i], vertices[j]))

    return edges

def enumerateAllGraphs(datasets):
    # use the reweighted dataset after fixing   
    reweighted_dataset = datasets[3]

    # uncomment the below code to test using full data set
    # reweighted_dataset = datasets[0]
    # reweighted_dataset.pop("R1")
    # reweighted_dataset.pop("R2")
    # print("using fully observed dataset")

    G = Graph(vertices=reweighted_dataset.columns)
    vertices = reweighted_dataset.columns

    # generate all the possible subset of edges from the dataset
    possible_edges = generateEdges(G.vertexList())

    G_star = None
    min_bic_score = None
    # generate all possible subsets of possible_edges
    for i in range(1, len(possible_edges)+1):
        subsets = findSubsets(possible_edges, i)
        # construct a graph for each subset and calculate its BIC score
        for subset in subsets:
            G_orig = Graph(vertices=vertices, edges=subset)
            # for each subset we also have to consider the graph where we flip
            # a subset of the edges
            for j in range(len(subset)+1):
                # find the subsets of this subset
                subsets_of_subset = findSubsets(subset, j)
                G_this = Graph(vertices=G_orig.vertexList(), edges=G_orig.edges())
                for subset_of_subset in subsets_of_subset:
                    # flip all edges in this subset
                    for edge in subset_of_subset:
                        G_this.delete_edge(edge[0], edge[1])
                        G_this.add_edge(edge[1], edge[0])

                    # make sure it's acylic and calculate its BIC score
                    this_bic = None
                    if acyclic(G_this) == True:
                        this_bic = bic_score(G_this, reweighted_dataset)

                    # check if this graph has the best BIC score
                    if this_bic != None and (min_bic_score == None or this_bic < min_bic_score):
                        min_bic_score = this_bic
                        G_star = G_this
    
    print(min_bic_score)
    print(G_star.edges())

    G_real = Graph(vertices=vertices, edges=(("Y1", "X1"), ("Y2", "X1"), ("X1", "X2"), ("Y1", "X2")))
    print(bic_score(G_real, reweighted_dataset))
    print(G_real.edges())

if __name__ == "__main__":
    np.random.seed(10)
    random.seed(10)

    datasets = testShadowYGraph(size=5000, verbose=False, datasets=True)
    G_star = causal_discovery(datasets[3], [("Y1", "X1"), ("Y1", "X2")], verbose=False)

    print(G_star.edges())

    # enumerateAllGraphs(datasets)
