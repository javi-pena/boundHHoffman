#!/usr/bin/env python
# coding: utf-8
"""
Created on Sat Feb  4 09:41:14 2023

@author: Javier Peña
"""

import cvxpy as cp
import numpy as np

def partition(A):
    # compute partition by solving (7)
    m,n = A.shape
    x = cp.Variable(n)
    y = cp.Variable(m)
    s = cp.Variable(m)
    t = cp.Variable(1)
    constraints = [np.ones(m)@y+np.ones(m)@s==1, y+s-t*np.ones(m)>=0,A@x+s==0,A.T@y == 0,y>=0, s>=0]        
    prob = cp.Problem(cp.Maximize(t),constraints)
    prob.solve(solver = cp.GUROBI)
    B = np.where(y.value>s.value)[0]
    N = np.where(s.value>y.value)[0]
    return B,N

def hoffmanN(A):
    # compute upper bound on H0(A) assuming A@x < 0 is feasible
    m,n = A.shape
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(x,np.identity(n))),[A@x >= np.ones(m)])
    prob.solve(solver = cp.GUROBI)
    return np.linalg.norm(x.value)


def hoffmanB(A):
    # compute upper bound on H0(A) assuming A.T@y=0, y > 0 is feasible
    m,n = A.shape
    y = cp.Variable(m)
    prob = cp.Problem(cp.Maximize(cp.sum(cp.log(y))),[A.T@y == 0, np.ones(m)@y==1, y >=0])
    prob.solve()
    M = A.T@np.diag(y.value)
    r = np.linalg.matrix_rank(M)
    _,s,_ = np.linalg.svd(M)
    smin = np.min(s[:r])
    return 2/smin

def hoffmanLK(AB,AN):
    # compute upper bound on H(L,K)
    Q,R = np.linalg.qr(AB.T,mode='complete')
    r = np.linalg.matrix_rank(AB)
    Q = Q[:,r:]
    d = ((AN**2)@np.ones(AN.shape[1]))**(0.5)
    M = np.diag(1/d)@AN@Q
    m,n = M.shape
    z = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(z,np.identity(n))),[M@z >= np.ones(m)])
    prob.solve(solver = cp.GUROBI)
    return 1+2*np.linalg.norm(z.value)

def boundH0(A):
    # Find partition and stitch together the bounds on H0(AB), H0(AN), H(L,K)
    B,N=partition(A)    
    if len(N)>0:
        AN = A[N,:]
        h0 = hoffmanN(AN)
    if len(B)>0:
        AB= A[B,:] 
        h1 = hoffmanB(AB)
    if len(N)==0:
        return h1
    if len(B)==0:
        return h0
    h2 = hoffmanLK(AB,AN)
    return h2*max(h0,h1)
