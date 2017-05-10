#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 21:11:30 2017

@author: phildec

test AIP4 : run batch of sudoku games
"""

import sys
import os

file_in=open("sudokus_start.txt")
file_res=open("sudokus_finish.txt")

line=file_in.readline()
line_res=file_res.readline()
n=0

while(line):
    n+=1
    print('.',end='')
    #print(line)
    os.system("python driver_3.py "+line)
    file_in2=open("output.txt")
    line2=file_in2.readline()
    #print("result:" +line2)
    if (line2==line_res): 
        print("--- MATCH! at %d" %n)
        print(line_res)
        print(line2)
    file_in2.close()
    
    line=file_in.readline()
    line_res=file_res.readline()


file_in.close()    
file_res.close()