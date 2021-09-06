# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 23:41:14 2021

@author: sunny"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style


class LMSalg(object):
    def _init_(self,w_init,input_mat):
        self.weights=w_init
        self.samples,columns=np.shape(input_mat)
    
    def _str_(self,input_mat,labels,f):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print('最终权向量为：')
        print(self.weights)
        result=input_mat*self.weights
        sum1=0
        sum2=0
        error=result-labels
        for i in range(0,self.samples):
            sum2=sum2+error[i,0]*error[i,0]
        print('总误差为：%0.4f'%sum2)
        result=f(result)
        for i in range (0,200):
            if(result[i]!=labels[i]):
                sum1=sum1+1
        print('错误分割点个数为：%d'%sum1)
        x = np.arange(-8, 8, 0.5)
        y = np.arange(-8, 8, 0.5)
        X, Y = np.meshgrid(x, y)
        Z1=self.weights[0,0]*X+self.weights[1,0]*Y+self.weights[2,0]
        Z2=f(self.weights[0,0]*X+self.weights[1,0]*Y+self.weights[2,0])
        ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=plt.cm.hot)
        ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=plt.cm.hot)
        ax.set_zlim(-2,2)
        plt.show()
        labels1=np.array(labels[0:100,:])
        labels2=np.array(labels[100:200,:])
        ax.scatter(np.array(input_mat[0:100,0]), np.array(input_mat[0:100,1]), zs=labels1, zdir="z", c="#00DDAA", marker="o", s=40)
        ax.scatter(np.array(input_mat[100:200,0]), np.array(input_mat[100:200,1]), zs=labels2, zdir="z", c="#FF5511", marker="^", s=40)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        style.use('ggplot')
        
        
              
        
        
    def train(self,input_mat,labels,rate):
        for i in range(0,10000):
            self.one_iteration(input_mat,labels,rate)
            
    def one_iteration(self,input_mat,labels,rate):
        position=np.random.randint(0,self.samples)
        subtraction=input_mat[position,:]*self.weights-labels[position,0] 
        subtraction1=rate*((np.mat(input_mat[position])).T)*subtraction
        self.weights=self.weights-subtraction1
        
class LMSalg1(LMSalg):
    def train(self,input_mat,labels,rate):
        for i in range(0,1000):
            self.one_iteration(input_mat,labels,rate)
            
    def one_iteration(self,input_mat,labels,rate):
        self.weights=self.weights-rate*input_mat.T*(input_mat*self.weights-labels)
    
class Sserr(LMSalg):

    def Optimal_solution(self,input_mat,labels):
        self.weights=input_mat.I*labels
    

def get_training_dataset_random(choose):
   
    labels=np.mat(np.ones((200,1)))
    for i in range (100,200):
        labels[i,0]=-1
        
    mean = (-5, 0)
    cov = [[1, 0], [0, 1]]
    x1 = np.random.multivariate_normal(mean, cov,100)
    mean = (5,0)            
    x2 = np.random.multivariate_normal(mean, cov, 100)
    input_mat=np.vstack((x1,x2))
    plt.scatter(x1[:,0],x1[:,1],s=50)
    plt.scatter(x2[:,0],x2[:,1],s=50)
    plt.show()
    style.use('ggplot')
    
    input_num,column=input_mat.shape
    add_mat=np.mat(np.ones((input_num,1)))
    input_mat=np.hstack((input_mat,add_mat))
    w_init=np.mat(np.zeros((column+1,1)))
    
    if choose==1:
        rate=float(input('请输入学习率:'))
        return input_mat,labels,w_init,rate
    else:
        return input_mat,labels,w_init

def f(x):
    if x>0: 
        return 1 
    else:
        return -1

def train_Linear_element():
    input_mat,labels,w_init,rate=get_training_dataset_random(1)
    p = LMSalg()
    p._init_(w_init,input_mat)
    p.train(input_mat,labels,rate)
    X = np.linspace( -8, 8, 100)
    Y =p.weights[0,0]/p.weights[1,0]*X-p.weights[2,0]/p.weights[1,0]
    plt.plot(X, Y, color="red", linewidth=1.0, linestyle="-") # 将100个散点连在一起
    plt.show()
    p._str_(input_mat,labels,np.vectorize(f))
    return p
def train_Linear_element1():
    input_mat,labels,w_init,rate=get_training_dataset_random(1)
    p = LMSalg1()
    p._init_(w_init,input_mat)
    p.train(input_mat,labels,rate)
    X = np.linspace( -3, 3, 100)
    Y =p.weights[0,0]/p.weights[1,0]*X-p.weights[2,0]/p.weights[1,0]
    plt.plot(X, Y, color="red", linewidth=1.0, linestyle="-") # 将100个散点连在一起
    plt.show()
    p._str_(input_mat,labels,np.vectorize(f))
    return p

def train_Linear_element2():
    input_mat,labels,w_init=get_training_dataset_random(0)
    p = Sserr()
    p._init_(w_init,input_mat)
    p.Optimal_solution(input_mat,labels)
    X = np.linspace(-3, 3, 100)
    Y =p.weights[0,0]/p.weights[1,0]*X-p.weights[2,0]/p.weights[1,0]
    plt.plot(X, Y, color="red", linewidth=1.0, linestyle="-") # 将100个散点连在一起
    plt.show()
    p._str_(input_mat,labels,np.vectorize(f))
    return p

_name_='main2'

if _name_=='main':
    Linear_element=train_Linear_element()
if _name_=='main1':
    Linear_element=train_Linear_element1()
if _name_=='main2':
    Linear_element=train_Linear_element2()    
    
    
