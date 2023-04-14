# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 10:54:05 2022

@author: wsb15
"""
import multiprocessing

count = 0

def f(g):
    global count
    count +=1
    print(count)

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    pool.map_async(f,[1,2,3,4,5])
    print(count)