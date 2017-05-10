# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:06:58 2017

@author: philippe.de-cuetos
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:33:11 2017

@author: philippe.de-cuetos

W9 Project: Constraint Satisfaction Problems
solving Sudoku
-  first implement the AC-3 algorithm
-  Now, implement backtracking using the minimum remaining value heuristic


First try with : 003020600900305001001806400008102900700000008006708200002609500800203009005010300
that should output : 483921657967345821251876493548132976729564138136798245372689514814253769695417382

then with : 000100702030950000001002003590000301020000070703000098800200100000085060605009000
that should output : 956138742237954816481672953594867321128593674763421598879246135312785469645319287

try : 956138742237954816481672953590000301020000070703000098800200100000085060605009000
"""
import sys
import string
import math
import copy 

letter_range=string.ascii_uppercase

def transToString(state,n):
    output=""
    for i in range(n):
        for j in range(n):
            term=letter_range[i]+str(j+1)
            if (len(state[term])>1): output+=str(0)
            else: output+=str(state[term][0])
    return(output)

def getNeighbors(n,X):
    neighbors=list()
    #in which squares is this box?
     
    i=letter_range.index(X[0])
    j=int(X[1])-1
          
    for k in range(n):
        neighbor=letter_range[i]+str(k+1) #same line 
        if neighbor!=X: neighbors.append(neighbor)
        neighbor =letter_range[k]+str(j+1) #same column
        if neighbor!=X: neighbors.append(neighbor)
    
    #neighbors withing same square :            
    #excluding same line and same column already added above
    square_n=int(math.sqrt(n)) 
    square_i=i//square_n
    square_j=j//square_n     
    
    for k in range(square_n):
        for l in range(square_n):
            line=square_i*square_n+k
            column=square_j*square_n+l+1
            if (line!=i)&(column!=j+1): 
                neighbor=letter_range[line]+str(column)
                neighbors.append(neighbor)
    
    return(neighbors)
    
    

def AC3(state,n):
    """ 
    il faut tout mettre dans l'état, et considérer les voisins des lignes
    colonnes et carrés de proche en proche pour enlever 
    les valeurs une à une 
    """
    #init state : dict with all possible values for each case
    arc_queue=list()
          
    for i in range(n):
        for j in range(n):
            term=letter_range[i]+str(j+1)
            #init queue with all arcs
            neighbors=getNeighbors(n,term)
            for neighbor in neighbors:
                arc_queue.append((term,neighbor))
            
           
    while(len(arc_queue)>0):
        Xi,Xj=arc_queue.pop(0)
        #revise :
        revised=False
        if (len(state[Xj])==1): #only case here where we can have no consistency
           li=len(state[Xi])
           i=0
           while(i<li):
               value=state[Xi][i] 
               if (value==state[Xj][0]): 
                   state[Xi].remove(value)
                   i-=1
                   li-=1
                   revised=True
               i+=1 
        if revised:
            if len(state[Xi])==0: return(False,state)
            for neigbor in getNeighbors(n,Xi):
                if neigbor!=Xj: arc_queue.append((neigbor,Xi))
                
    
    return(True,state)

def checkComplete(state,n):
    for i in range(n):
        for j in range(n):
            term=letter_range[i]+str(j+1)
            value=state[term]
            if len(value)>1: return(False)    
    return(True)

def selectMinRemaining(state,n):
    min_nb=9
    min_term='A1'
    for i in range(n):
        for j in range(n):
            term=letter_range[i]+str(j+1)
            value=state[term]
            if (len(value)>1)&(len(value)<min_nb): 
                min_nb=len(value)
                min_term=term
            if (min_nb==2): return(min_term)
    return(min_term)
    

def backtracking_search(csp,n):
    return(backtrack(csp,n))

def backtrack(assignment, n):
    if checkComplete(assignment,n): 
        #print("WIIIIIIIIIIINNNNNN")
        return(True,assignment)    
    #select unassigned variables : use minimum remaining value
    var=selectMinRemaining(assignment,n)
    print(var,assignment[var])
    for value in assignment[var]:
        #print("--try : %s=%d for " %(var,value),transToString(assignment,n))
        #input("press key:")
        #print("A5:",assignment["A5"])
        #check consistency of this assignement with neighbors
        consistency=True
        neighbors=getNeighbors(n,var)
        for neighbor in neighbors:
            thelist=assignment[neighbor]
            if (len(thelist)==1)&(thelist[0]==value):
                consistency=False
                #print("not consistent")
                break
        
        if consistency:
            #add var=value to assignment i.e. copy into new restrained dict ??
            # ... or keep changes and reapply them later ?
            #newassign={}
            newassign=copy.deepcopy(assignment)
            newassign[var]=[value]
            #do inference 
            inferRes,newassign = AC3(newassign,n)
            
            if (inferRes): 
                #return(backtrack(newassign,n))
                result,newassign=backtrack(newassign,n)
                if result:
                #if checkComplete(newassign,n): 
                    #print("WIIIIIIIIIIINNNNNN")
                    return(True,newassign) 
            #else: 
                #print("AC3 failed") 
                #return(False,{})  
    return(False,assignment)
                     
            
#main program
if len(sys.argv)!=2:
    sys.stderr.write("Usage : python %s sudoku_sequence\n" %sys.argv[0])
    raise SystemExit(1)

seq = sys.argv[1]
#nb of lines and columns
n=int(math.sqrt(len(seq)))

#write into sudoku board structure
sudoku={}

for i in range(n):
    for j in range(n):
        case=letter_range[i]+str(j+1)
        #sudoku.update({case:int(seq[i*n+j])})
        value=int(seq[i*n+j])
        if value==0: sudoku[case]=list(range(1,n+1))
        else: sudoku[case]=[value]

#AC-3 algo
isAC3,state = AC3(sudoku,n)
result,final=backtracking_search(state,n)
#final=state
#result=isAC3

#print resulting state as string and output to file
file_out=open("output.txt","w")

if not result: print("failed : No possible solution")
else:
    output=transToString(final,n)
    print(output)
    print(output,file=file_out)
        
file_out.close()
                





