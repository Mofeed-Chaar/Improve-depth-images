#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 13:44:54 2024

@author: chaar
"""
test = {'1':1,'2':2,'3':3,'4':4,'5':5}

print(list(test.keys()))

print(test[list(test.keys())[0]])

print(len(test))
for key in test.keys():
    print(test[key])