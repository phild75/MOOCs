#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:58:28 2017

@author: phildec

python driver.py <method> <board>

bfs (Breadth-First Search)
dfs (Depth-First Search)
ast (A-Star Search)
ida (IDA-Star Search)

ex :  driver.py bfs 0,8,7,6,5,4,3,2,1

should output file :
path_to_goal: ['Up', 'Left', 'Left']
cost_of_path: 3
nodes_expanded: 10
fringe_size: 11
max_fringe_size: 12
search_depth: 3
max_search_depth: 4
running_time: 0.00188088
max_ram_usage: 0.07812500

"""

#IA course Week 2 : n-puzzle
import sys
import math
import numpy as np
import time
import psutil #to get ram usage

#structure du puzzle 
class Puzzle(object):
    
    """def __init__(self):
        self.size=1
        self.n=1
        
        #self.parentIndex=0 #index of parent in the order of explored puzzles

        self.board=np.zeros((self.n,self.n),dtype=int)
    """
    def addBoard(self,board_list):
        self.size=len(board_list)
        self.n=int(math.sqrt(self.size))
        
        self.board=np.zeros((self.n,self.n),dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                self.board[i,j]=board_list[i*self.n+j]
                
    def copy(self): 
        #copy puzzle into a new puzzle 
        #il doit y avoir une fonction de array + rapide ...
        newPuzzle=Puzzle()
        newPuzzle.n=self.n
        #newPuzzle=np.copy(self.board)
        newPuzzle.board=np.zeros((self.n,self.n),dtype=int)
        newPuzzle.n=self.n
        for i in range(self.n):
            for j in range(self.n):
                newPuzzle.board[i,j]=self.board[i,j]
                
        #newPuzzle.originPath=originPath
        #newPuzzle.parentIndex=parentIndex
        return(newPuzzle)
    
    def isequal(self,bPuzzle):
        for i in range(self.n):
            for j in range(self.n):
                if (bPuzzle.board[i,j]!=self.board[i,j]):
                    return False
        return True
    
    def computeManhattanDist(self):
        targetPuzzle=Puzzle()
        
        targetPuzzle.addBoard(list(range(self.n*self.n))) 
            
        dist=0
        for i in range(self.n):
            for j in range(self.n):
                target=self.board[i,j] #chiffre dont la distance est à calculer
                #position dans la matrice d'arrivée 
                if target>0: 
                    index_target=np.where(targetPuzzle.board==target) 
                    i_target=int(index_target[0])
                    j_target=int(index_target[1])
                    dist+=abs(i-i_target)+abs(j-j_target)
        return(dist)    
    
    def print(self):
        print(self.board)
        
        
#état du puzzle
class State(object):
    
    def __init__(self,index=0): 
        self.neighbors=list()
        self.depth=0 #depth in the tree => g(n) in A-STAR
        self.originPath='None'
        self.parent_state=self
        self.n=0
        self.h_n=0
        
    def addPuzzle(self,puzzle):
         self.puzzle=puzzle
         self.n=puzzle.n
         self.h_n=puzzle.computeManhattanDist()   #heuristic fct (Manhattan) h(n) in A-STAR
                      
    def expand(self):
        #ordre UDLR par convention : ajouter seulement les fils différents du père         
        theNeighbor=self.addUp()
        if (theNeighbor):
            self.neighbors.append(self.addUp())
        
        theNeighbor=self.addDown()
        if (theNeighbor):
            self.neighbors.append(self.addDown())
        
        theNeighbor=self.addLeft()
        if (theNeighbor):
            self.neighbors.append(self.addLeft())
        
        theNeighbor=self.addRight()
        if (theNeighbor):
            self.neighbors.append(self.addRight())
        
        return(self.neighbors)
        
    def addUp(self):
        #l'index du puzzle courant est gardé en copie dans son puzzle fils
        child=State()
        child.puzzle=self.puzzle.copy() #copy puzzle
        child.n=child.puzzle.n
        child.depth=self.depth+1
        child.originPath='Up'
        child.parent_state=self
        
        index_0=np.where(child.puzzle.board==0) #returns index of 0 but as a ndarray of dim 1 
        i_0=int(index_0[0])
        j_0=int(index_0[1])
             
        #print(i_0,j_0)
        if i_0>0: #déjà à la 1ère ligne => on renvoie le même puzzle
            #permute 0 et le nb juste au-dessus
            child.puzzle.board[i_0,j_0]=child.puzzle.board[i_0-1,j_0]
            child.puzzle.board[i_0-1,j_0]=0
            #print(child.puzzle.board)
            child.h_n=child.puzzle.computeManhattanDist()
            #print("Manhattan:", child.h_n)
            return(child)
        else:
            child.h_n=self.h_n #no change in the board => no change in h(n)
            return
        
    def addDown(self):
        child=State()
        child.puzzle=self.puzzle.copy() #copy puzzle
        child.n=child.puzzle.n
        child.depth=self.depth+1
        child.originPath='Down'
        child.parent_state=self
        
        index_0=np.where(child.puzzle.board==0) #returns index of 0 but as a ndarray of dim 1 
        i_0=int(index_0[0])
        j_0=int(index_0[1])
        
        if i_0<self.n-1: #déjà à la dernière ligne => on renvoie le même puzzle
            #permute 0 et le nb juste en-dessous
            child.puzzle.board[i_0,j_0]=child.puzzle.board[i_0+1,j_0]
            child.puzzle.board[i_0+1,j_0]=0
            child.h_n=child.puzzle.computeManhattanDist()
            return(child)
        else:
            child.h_n=self.h_n
            return
    
    def addLeft(self):
        child=State()
        child.puzzle=self.puzzle.copy() #copy puzzle
        child.n=child.puzzle.n
        child.depth=self.depth+1
        child.originPath='Left'
        child.parent_state=self
        
        index_0=np.where(child.puzzle.board==0) #returns index of 0 but as a ndarray of dim 1 
        i_0=int(index_0[0])
        j_0=int(index_0[1])
        
        if j_0>0: #déjà à la 1ère colonne => on renvoie le même puzzle
            #permute 0 et le nb juste à gauche
            child.puzzle.board[i_0,j_0]=child.puzzle.board[i_0,j_0-1]
            child.puzzle.board[i_0,j_0-1]=0
            child.h_n=child.puzzle.computeManhattanDist()
            return(child)
        else:
            child.h_n=self.h_n
            return
    
    def addRight(self):
        child=State()
        child.puzzle=self.puzzle.copy() #copy puzzle
        child.n=child.puzzle.n
        child.depth=self.depth+1
        child.originPath='Right'
        child.parent_state=self
        
        index_0=np.where(child.puzzle.board==0) #returns index of 0 but as a ndarray of dim 1 
        i_0=int(index_0[0])
        j_0=int(index_0[1])
        
        if j_0<self.n-1: #déjà à la dernière colonne => on renvoie le même puzzle
            #permute 0 et le nb juste à droite
            child.puzzle.board[i_0,j_0]=child.puzzle.board[i_0,j_0+1]
            child.puzzle.board[i_0,j_0+1]=0
            child.h_n=child.puzzle.computeManhattanDist()
            return(child)
        else:
            child.h_n=self.h_n
            return
        
def bfs(state_init,state_goal, out_params):
    """
    frontier = Queue.new(initialState) // different
     explored = Set.new()       

     while not frontier.isEmpty():
          state = frontier.dequeue()  //different   
          explored.add(state)      
          if goalTest(state):
               return SUCCESS(state)
          for neighbor in state.neighbors():
               if neighbor not in frontier U explored:  
                    frontier.enqueue(neighbor) //different   
     return FAILURE


    """
    
    #init frontier and explored list (can also use sets for explored ?)
    frontier=list()
    explored=list()
    
    #variables to output
    completePath=list() 
    directPath=list()
    maxdepth=0
    nodes_expanded=0 
    max_fringe_size=0              
             
    frontier.append(state_init)
    
    while len(frontier)!=0:
        current_state=frontier.pop(0) #FIFO : remove elt first added to the list
        
        if current_state.originPath!='None':#don't add origin puzzle to path
            completePath.append(current_state.originPath)
          
        explored.append(current_state)    
        #exploredIndex=len(explored)-1 #index of current puzzle in list of explored
            
        if current_state.puzzle.isequal(state_goal.puzzle):
            #print("final board reached:")
            #current_state.puzzle.print()
            #print("path complet : ",completePath)
            
            #retrieve direct path en remontant de parent en parent
            while current_state.depth>0:
                directPath.append(current_state.originPath)
                current_state=current_state.parent_state
            directPath.reverse()
            
            out_params["path_to_goal"]=directPath
            out_params["cost_of_path"]=len(directPath)
            out_params["nodes_expanded"]=nodes_expanded 
            out_params["fringe_size"]=len(frontier)
            out_params["max_fringe_size"]=max_fringe_size 
            out_params["search_depth"]=len(directPath) #à voir pour les autres algos
            out_params["max_search_depth"]=maxdepth 
            return
        #print("new explored:",exploredIndex)
        #crée un nouvel état à partir du puzzle pour trouver les voisins
        #state_current=State(puzzle_current,exploredIndex) 
        
        neighbors=current_state.expand() 
   
        if len(neighbors)>0: 
            nodes_expanded+=1 #nodes that have been expanded to children nodes
            
                
        for i in range(len(neighbors)):
            #print("neighbor %d:" %i)
            #print(neighbors[i].puzzle)
            neighbors_current=neighbors[i]
            #check if not in frontier and not in explored 
            addOK=True
            for j in range(len(frontier)):
                if (neighbors_current.puzzle.isequal(frontier[j].puzzle)):
                    addOK=False
                    break
            if (addOK):
                for j in range(len(explored)):
                    if (neighbors_current.puzzle.isequal(explored[j].puzzle)):
                        addOK=False
                        break
            if (addOK):
                frontier.append(neighbors_current)
                
                if neighbors_current.depth>maxdepth:
                    maxdepth=neighbors_current.depth
                    #print("maxdepth= ",maxdepth)
                    
                if len(frontier)>max_fringe_size:
                    max_fringe_size=len(frontier)
        
    #print("final board not reached:")
    #puzzle_current.print()
    return


def dfs(state_init,state_goal,out_params):
    """
    function DEPTH-FIRST-SEARCH(initialSate, goalTest)     
     returns SUCCESS or FAILURE :

     frontier=Stack.new(initialState) //different
     explored = Set.new()       

     while not frontier.isEmpty():
          state = frontier.pop()     // different    
          explored.add(state)     
          if goalTest(state):
               return SUCCESS(state)
          for neighbor in state.neighbors():
               if neighbor not in frontier U explored:  
                    frontier.push(neighbor)      // different 
     return FAILURE

    """
    #init frontier and explored list (can also use sets for explored ?)
    frontier=list()
    explored=list()
    explored_frontier_comp=set() #put the list of explored or frontier tabs as string for easy compare
    
    #variables to output
    completePath=list() 
    directPath=list()
    maxdepth=0
    nodes_expanded=0 
    max_fringe_size=0              
             
    frontier.append(state_init)
    
    while len(frontier)!=0:
        current_state=frontier.pop() #LIFO : remove elt first added to the list
        
        if current_state.originPath!='None':#don't add origin puzzle to path
            completePath.append(current_state.originPath)
          
        explored.append(current_state)
        explored_frontier_comp.add(current_state.puzzle.board.tostring())
        #exploredIndex=len(explored)-1 #index of current puzzle in list of explored
            
        if current_state.puzzle.isequal(state_goal.puzzle):
            #print("final board reached:")
            #current_state.puzzle.print()
            #print("path complet : ",completePath)
            
            #retrieve direct path en remontant de parent en parent
            while current_state.depth>0:
                directPath.append(current_state.originPath)
                current_state=current_state.parent_state
            directPath.reverse()
            
            out_params["path_to_goal"]=directPath
            out_params["cost_of_path"]=len(directPath)
            out_params["nodes_expanded"]=nodes_expanded 
            out_params["fringe_size"]=len(frontier)
            out_params["max_fringe_size"]=max_fringe_size 
            out_params["search_depth"]=len(directPath) #à voir pour les autres algos
            out_params["max_search_depth"]=maxdepth 
            return
        #print("new explored:",exploredIndex)
        #crée un nouvel état à partir du puzzle pour trouver les voisins
        #state_current=State(puzzle_current,exploredIndex) 
        
        neighbors=current_state.expand() 
   
        if len(neighbors)>0: 
            nodes_expanded+=1 #nodes that have been expanded to children nodes
            """if neighbors[0].depth>maxdepth:
                maxdepth=neighbors[0].depth
                print("maxdepth= ",maxdepth)
               """ 
        for i in range(len(neighbors)):
            #print("neighbor %d:" %i)
            #print(neighbors[i].puzzle)
            neighbors_current=neighbors[len(neighbors)-i-1] #put reverse UDLR so that frontier is RLDU and respect order of pop
            #check if not in frontier and not in explored 
            strneighbor=neighbors_current.puzzle.board.tostring()
            if not(strneighbor in explored_frontier_comp):     
                frontier.append(neighbors_current)
                explored_frontier_comp.add(neighbors_current.puzzle.board.tostring())
                
                if neighbors_current.depth>maxdepth:
                    maxdepth=neighbors_current.depth
                    #print("maxdepth= ",maxdepth)
                
                if len(frontier)>max_fringe_size:
                    max_fringe_size=len(frontier)
    
    return

#used for heap implementation (sort function)
def getKey(item):
    return item[0]

def ast(state_init,state_goal, out_params):
    """
    function A-STAR-SEARCH(initialSate, goalTest)     
     returns SUCCESS or FAILURE : //cost fct f(n) = g(n) + h(n) : different

     frontier = Heap.new(initialState) 
     explored = Set.new()       

     while not frontier.isEmpty():
          state = frontier.deleteMin() 
          explored.add(state)      

          if goalTest(state):
               return SUCCESS(state)

          for neighbor in state.neighbors():
               if neighbor not in frontier U explored:  
                    frontier.insert(neighbor) 
               else if neighbor in frontier:                    
                    frontier.decreaseKey(neighbor)         
     return FAILURE

    """
    #dictionary for frontier would be faster ? 
    frontier=list() #now list of tuple (f(n),state) ordered by f(n)=h(n)+g(n) 
    explored=list()
    explored_frontier_comp=set() #put the list of explored or frontier tabs as string for easy compare
    frontier_comp=set() #put the list of frontier tabs as string for easy compare

    #variables to output
    completePath=list() 
    directPath=list()
    maxdepth=0
    nodes_expanded=0 
    max_fringe_size=0              
             
    frontier.append((0,state_init))
    
    while len(frontier)!=0:
        frontier.sort(key=getKey)   #order stak by 1st element
        #print(frontier)
        j=0
        if len(frontier)>1:
            for i in range(len(frontier)-1):
                if frontier[i+1][0]>frontier[i][0]: #get 1st element with different f_n
                    break
                for j in range(i): #respect UDLR order 
                    if frontier[j][1].originPath=='Up':
                        break
                    if frontier[j][1].originPath=='Down':
                        break
                    if frontier[j][1].originPath=='Left':
                        break
        current_state=frontier.pop(j)[1] 
        
        #print("current board h_n =", current_state.h_n)
        #current_state.puzzle.print()                          
                                  
        if current_state.originPath!='None':#don't add origin puzzle to path
            completePath.append(current_state.originPath)
          
        explored.append(current_state)
        explored_frontier_comp.add(current_state.puzzle.board.tostring())

        #exploredIndex=len(explored)-1 #index of current puzzle in list of explored
            
        if current_state.puzzle.isequal(state_goal.puzzle):
            #print("final board reached:")
            #current_state.puzzle.print()
            #print("path complet : ",completePath)
            
            #retrieve direct path en remontant de parent en parent
            while current_state.depth>0:
                directPath.append(current_state.originPath)
                current_state=current_state.parent_state
            directPath.reverse()
            
            out_params["path_to_goal"]=directPath
            out_params["cost_of_path"]=len(directPath)
            out_params["nodes_expanded"]=nodes_expanded 
            out_params["fringe_size"]=len(frontier)
            out_params["max_fringe_size"]=max_fringe_size 
            out_params["search_depth"]=len(directPath) #à voir pour les autres algos
            out_params["max_search_depth"]=maxdepth 
            return
        #print("new explored:",exploredIndex)
        #crée un nouvel état à partir du puzzle pour trouver les voisins
        #state_current=State(puzzle_current,exploredIndex) 
        
        neighbors=current_state.expand() 
   
        if len(neighbors)>0: 
            nodes_expanded+=1 #nodes that have been expanded to children nodes
 
        for i in range(len(neighbors)):
            #print("neighbor %d:" %i)
            #print(neighbors[i].puzzle)
            neighbors_current=neighbors[len(neighbors)-i-1] #put reverse UDLR so that frontier is RLDU and respect order of pop
            #check if not in frontier and not in explored 
            strneighbor=neighbors_current.puzzle.board.tostring()
            if not(strneighbor in explored_frontier_comp):     
                #put into frontier while respecting ascending order of f(n)
                f_n= neighbors_current.depth+neighbors_current.h_n
                frontier.append((f_n,neighbors_current))
                
                explored_frontier_comp.add(neighbors_current.puzzle.board.tostring())
                frontier_comp.add(neighbors_current.puzzle.board.tostring())
                
                if neighbors_current.depth>maxdepth:
                    maxdepth=neighbors_current.depth
                    #print("maxdepth= ",maxdepth)
                
                if len(frontier)>max_fringe_size:
                    max_fringe_size=len(frontier)
            
            elif strneighbor in frontier_comp: #if node already in frontier, decrease key
                f_n= neighbors_current.depth+neighbors_current.h_n
                #retrieve the state already in frontier and update its f_n
                #first get index of state in frontier that has the same board as neighbors_current
                for i in range(len(frontier)):
                    if neighbors_current.puzzle.isequal(frontier[i][1].puzzle):
                        break
                
                #then if its f_n bigger than neighbors_current, change with neighbors_current
                if f_n < frontier[i][0]:
                    frontier.pop(i)
                    frontier.append((f_n,neighbors_current))
                
    return

def ida(state_init,state_goal, out_params):
    """
    As before, for the choice of heuristic, use the Manhattan priority function. 
    Recall from lecture that implementing the Iterative Deepening Search (IDS) 
    algorithm involves first implementing the Depth-Limited Search (DLS) algorithm
    as a subroutine. Similarly, implementing the IDA-Star Search algorithm 
    involves first implementing a modified version of the DLS algorithm 
    that uses the heuristic function in addition to node depth.
         
    """
    
    cost_limit=1 #initial cost limit for 1st iteration
    cost_step=1 #increase in cost limite for each iteration
    
    while(True):
        #print("-------------cost_limit : ",cost_limit)
        
        #init frontier and explored list (can also use sets for explored ?)
        frontier=list()
        explored=list()
        explored_frontier_comp=set() #put the list of explored or frontier tabs as string for easy compare
    
        #variables to output
        completePath=list() 
        directPath=list()
        maxdepth=0
        nodes_expanded=0 
        max_fringe_size=0              
             
        frontier.append(state_init)
    
        while (len(frontier)!=0):
            current_state=frontier.pop()#LIFO : remove elt first added to the list 
            if current_state.originPath!='None':#don't add origin puzzle to path
                completePath.append(current_state.originPath)
          
            explored.append(current_state)
            explored_frontier_comp.add(current_state.puzzle.board.tostring())
            
            
            if current_state.puzzle.isequal(state_goal.puzzle):
                #print("final board reached:")
                current_state.puzzle.print()
                
                #retrieve direct path en remontant de parent en parent
                while current_state.depth>0:
                    directPath.append(current_state.originPath)
                    current_state=current_state.parent_state
                directPath.reverse()
            
                out_params["path_to_goal"]=directPath
                out_params["cost_of_path"]=len(directPath)
                out_params["nodes_expanded"]=nodes_expanded 
                out_params["fringe_size"]=len(frontier)
                out_params["max_fringe_size"]=max_fringe_size 
                out_params["search_depth"]=len(directPath) #à voir pour les autres algos
                out_params["max_search_depth"]=maxdepth 
                return
        
            neighbors=current_state.expand() 
   
            if len(neighbors)>0: 
                nodes_expanded+=1 #nodes that have been expanded to children nodes
     
            for i in range(len(neighbors)):
                #print("neighbor %d:" %i)
                #print(neighbors[i].puzzle)
                neighbors_current=neighbors[len(neighbors)-i-1] #put reverse UDLR so that frontier is RLDU and respect order of pop
                #check if not in frontier and not in explored 
                strneighbor=neighbors_current.puzzle.board.tostring()
                
                f_n=neighbors_current.depth+neighbors_current.h_n 
                if (not(strneighbor in explored_frontier_comp))and(f_n<=cost_limit):     
                    frontier.append(neighbors_current)
                    explored_frontier_comp.add(neighbors_current.puzzle.board.tostring())
                
                    if neighbors_current.depth>maxdepth:
                        maxdepth=neighbors_current.depth
                        #print("maxdepth= ",maxdepth)
                
                    if len(frontier)>max_fringe_size:
                        max_fringe_size=len(frontier)
        #reached end of frontier nodes that have f_n<cost_limit
        cost_limit+=cost_step 
        
    return

#main program 
start_time=time.time()

if len(sys.argv)!=3:
    sys.stderr.write("Usage : python %s <bfs,dfs,ast,ida> board\n" %sys.argv[0])
    raise SystemExit(1)

method=sys.argv[1]
s=sys.argv[2]

#convert board as a 1-dimension list of integer elements 
s=s.split(',')
board=list()
for i in range(len(s)):
    board.append(int(s[i]))

#create init and goal states
puzzle_start=Puzzle() #Puzzle représentant l'état initial
puzzle_start.addBoard(board)
print("initial board:")
puzzle_start.print()
init_state=State()
init_state.addPuzzle(puzzle_start)

puzzle_end=Puzzle()
puzzle_end.addBoard(list(range(puzzle_start.size)))
goal_state=State()
goal_state.addPuzzle(puzzle_end)

#create dictionary for parameters changed by the used algo
out_params=dict(path_to_goal=0,cost_of_path=0,nodes_expanded=0,fringe_size=0,max_fringe_size=0,search_depth=0,max_search_depth=0)

if(method=="bfs"):
    bfs(init_state, goal_state, out_params)
elif(method=="dfs"):
    dfs(init_state, goal_state, out_params)
elif(method=="ast"):
    ast(init_state, goal_state, out_params)
else:
    ida(init_state, goal_state, out_params)

running_time=time.time()-start_time

process=psutil.Process()
max_ram_usage=process.memory_info().rss/1024/1024 #RAM in Mbytes
                            
#print results to file
file_out = open("output.txt","w")

print("path_to_goal:", out_params["path_to_goal"], file=file_out)
print("cost_of_path:", out_params["cost_of_path"], file=file_out)
print("nodes_expanded:", out_params["nodes_expanded"], file=file_out)
print("fringe_size:", out_params["fringe_size"], file=file_out)
print("max_fringe_size:", out_params["max_fringe_size"], file=file_out)
print("search_depth:", out_params["search_depth"], file=file_out)
print("max_search_depth:", out_params["max_search_depth"], file=file_out)
print("running_time: %.8f" %running_time, file=file_out)
print("max_ram_usage: %.8f" %max_ram_usage, file=file_out)

file_out.close()

print("path_to_goal:", out_params["path_to_goal"])
print("cost_of_path:", out_params["cost_of_path"])
print("nodes_expanded:", out_params["nodes_expanded"])
print("fringe_size:", out_params["fringe_size"])
print("max_fringe_size:", out_params["max_fringe_size"])
print("search_depth:", out_params["search_depth"])
print("max_search_depth:", out_params["max_search_depth"])
print("running_time: %.8f" %running_time)
print("max_ram_usage: %.8f" %max_ram_usage)