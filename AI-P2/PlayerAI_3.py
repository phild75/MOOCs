#Player AI : writable file !!
"""
instructions  :
    - employ minimax algo
    - with alpha-beta pruning (best reordering ?)
    - use heuristic functions + weighting (meta-algo to iterate over the space of weight vector /
    and get the best weights ?)
        - consider both qualitive and quantitative measures such as :
            - the absolute value of tiles,
            - the difference in value between adjacent tiles,
            - the potential for merging of similar tiles,
            - the ordering of tiles across rows, columns, and diagonals,
        => check : http://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048

Submissions for which the average maximum tile value falls halfway /
between 1024 and 2048 will receive full credit.

    + implement time limit 

"""
from BaseAI_3 import BaseAI
import time

timeLimit = 0.08
INF = 1000000 #infinity 

def timeOK(start_time):
    if (time.clock()-start_time>timeLimit): return(False)
    return(True)

def printGrid(grid):
    for k in grid.map:
        print(k)

class State:
    
    def __init__(self, initgrid, parent=None):
        self.grid=initgrid
        #self.computeHeuritic()
        self.parent_state=parent
        if parent==None: #maybe no need to know parent of each state in case of recursive fct
            self.depth=1 
        else: 
            self.depth=self.parent_state.depth+1
        self.originMove=-1
        
    def computeHeuristic(self):
        #1st very basic: nb of empty tiles
        empty_cells = self.grid.getAvailableCells()
        self.h_utility=len(empty_cells)
        return
    
    def expandPlayer(self): #case of PLAYER 
        self.children_states=[]
        moves = self.grid.getAvailableMoves()
        
        for i in range(len(moves)):
            
            child_grid = self.grid.clone()
            child_grid.move(moves[i])
            
            child_state=State(child_grid,self)
            child_state.originMove=moves[i]            
            self.children_states.append(child_state)
        
        return(self.children_states)
    
    def expandComputer(self):
        self.children_states=[]
        
        empty_cells = self.grid.getAvailableCells()
        #nodes order : 2,4,2,4,2,4,... for each empty cell
        for i in range(len(empty_cells)):
            for j in [2,4]: 
                child_grid=self.grid.clone()
                child_grid.setCellValue(empty_cells[i],j)
                child_state=State(child_grid,self)
                self.children_states.append(child_state)
        
        return(self.children_states)
    
    
#def getUtility(state): #for sorting state lists according to utility values
#    return(state.h_utility)

class PlayerAI(BaseAI):
    
    def getMove(self, grid):
        
        start_t = time.clock()
        
        #self.explored_frontier_comp=set()
        depth=1 #tree starts at depth = 1
        best_move=0 #Up by default
       
        while(timeOK(start_t)):
            #for now start from root at each new IDS turn, start at depth limit=2
            depth+=1 
            print("depth = ",depth)
            state_init=State(grid)
            #self.explored=list() #not need of explored because no risk of tree loop ?
            self.frontier=list()
            self.frontier.append(state_init)
            #implement IDS for current depth limit       
            self.advanceIDS(depth,start_t)
           
            #implement minimax algo
            if timeOK(start_t):
                print("minimax")
                child,best_value=self.maximize(state_init,depth,start_t,-INF,INF)
            if timeOK(start_t):
                best_move=child.originMove 
                print("... best_move:",best_move)

        return best_move #returns last best-move before timeLimitReached if any      

  

    def advanceIDS(self,depth_limit,start_t):
        #compute IDS until depth_limit
        #Remark : cannot loop in this game => no need to check node before getting from frontier
        t=time.clock()

        while len(self.frontier) and timeOK(start_t):  

            #print("frontier length:", len(self.frontier))    
            #print("frontier elt depth:", self.frontier[0].depth)  
            current_state=self.frontier.pop()#LIFO : remove elt first added to the list 

            #self.explored.append(current_state)
            if(current_state.depth<=depth_limit):     
                if(current_state.depth%2==1): #uneven depth => player turn
                    children=current_state.expandPlayer()

                    for i in range(len(children)):                     
                #put in reverse to maintain LIFO
                        self.frontier.append(children[len(children)-1-i])
            
                else: #computer turn
                    children=current_state.expandComputer()

                    for i in range(len(children)):                     
                        #put in reverse to maintain LIFO
                        self.frontier.append(children[len(children)-1-i])
        
        print("time taken for IDS: ",time.clock()-t)                
        return
 
       
       
    def maximize(self,state,depth_limit,start_time,alphap,betap):

        #print("MAX turn state_depth =",state.depth)        
        alpha=alphap
        beta=betap
        
        if (state.depth==depth_limit): #terminal node 
            state.computeHeuristic()
            #if (state.depth>=3): 
            #    print("----- max depth node for MAX : %d ------" %state.depth)
            return(None, state.h_utility)
        
        maxUtility=-INF
        maxChild=None
        #children=state.expandPlayer() 
         
        for i in range(len(state.children_states)):
           self.minimize(state.children_states[i],depth_limit,start_time,alpha,beta)
 
           print("MAX node utility : ",state.children_states[i].h_utility)               

           if (state.children_states[i].h_utility > maxUtility): #search for node with max utility
                maxChild=state.children_states[i]
                maxUtility=maxChild.h_utility
            
            
            #if (maxUtility >= beta): 
                #print("Pruning MAX at depth :",state.depth)
            #   break #cut search
            
            #if (maxUtility > alpha): 
            #   alpha=maxUtility

            
            
           if not timeOK(start_time): 
                break
            
        state.h_utility=maxUtility
                         
        return(maxChild, maxUtility)
    
    
    
    def minimize(self,state,depth_limit,start_time,alphap,betap):
   
        #print("MIN turn state_depth =",state.depth)  
        alpha=alphap
        beta=betap
        
        if (state.depth==depth_limit): #terminal node 
            state.computeHeuristic()
            #if state.depth>=3: 
            #    print("----- max depth node for MIN : %d ------" %state.depth)
            return(None, state.h_utility)
        
        minUtility=INF 
        minChild=None
        #children=state.expandComputer() 
         
        children = state.children_states
        for i in range(len(children)):
       
            self.maximize(children[i],depth_limit,start_time,alpha,beta)

            print("For i = %d MIN node utility %d " %(i,children[i].h_utility))
            #for k in children[i].grid.map:
            #    print(k)

            if (children[i].h_utility < minUtility):
                minChild=children[i]
                minUtility=minChild.h_utility
                
            #if (minUtility <= alpha): 
                #print("Pruning MIN at depth :", state.depth)
             #   break #cut search
            
            #if (minUtility < beta): 
            #    beta=minUtility
                
            if not timeOK(start_time): 
                break
        
        state.h_utility=minUtility
        
        return(minChild, minUtility)
        
"""
 
 function DECISION(state)   //main fct : find the child state with the highest utility value
    returns STATE:

    <child,_> = MAXIMIZE(state)
    return child     //returns the child with highest utility for MAX after recursing algo is done 

function MAXIMIZE(state)
    returns TUPLE of <STATE, UTILITY>:

    if TERMINAL-TEST(state):
        return <NULL,EVAL(state)>
    
    <maxChild,maxUtility> = <NULL,-inf>
    
    for child in state.children():     
        <_,utility>=MINIMIZE(child)  //give turn to MIN

        if utility > maxUtility:
            <maxChild,maxUtility> = <child,utility> //search for node with highest utility 

    return <maxChild, maxUtility>

function MINIMIZE(state)
    returns TUPLE of <STATE, UTILITY>:

    if TERMINAL-TEST(state):
        return <NULL,EVAL(state)>
    
    <minChild,minUtility>=<NULL,+inf>
    
    for child in state.children():
        <_,utility>=MAXIMIZE(child)  //give turn to MAX

        if utility < minUtility:
            <minChild,minUtility> = <child,utility> //search for node with lowest utility 

    return <minChild, minUtility>
 
"""