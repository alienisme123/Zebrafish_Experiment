from itertools import combinations
from tkinter import Y
import numpy as np
import matplotlib.pyplot as plt


class zbf:
    # path/number of animals/dctm = digtal to cm
    def __init__(self, path, noa, dtcm, dt=1/30):
        self.noa = noa
        self.tra = _traload(path, noa)
        self.frame = len(self.tra)
        self.dtcm = dtcm
        self.dt = dt
        self.vec = np.empty((self.frame-1,self.noa,2))       #vector    time/number of animal/dimension
        self.vec_s= np.empty((self.frame-1,self.noa,2))      #vector sum
        self.vec_rms = np.empty((self.frame-1,self.noa))     #the rms of vector sum 
        
        
    def getra(self):
        return self.tra

    
    # plot all fish tragetory
    def zbfplt(self):

        plt.figure(figsize=(6, 6))
        plt.title(str(self.noa)+" fishes trajetory")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Distance (cm)")

        for i in range(0, self.noa):
            plt.plot(self.tra[:, i, 0]/self.dtcm, self.tra[:, i, 1]/self.dtcm)

    # one dimemsion vs time graph, dim=0 is x axis,
    def oned_to_time(self, dim=0):
        time = np.linspace(0, len(self.tra)*self.dt, len(self.tra))
        axis = "Y" if dim else "X"
        for i in range(0, self.noa):
            plt.figure(i+self.noa*dim)
            plt.plot(time, self.tra[:, i, dim]/self.dtcm)
            plt.title(axis+"-time "+str(i+1)+" fish")
            plt.xlabel("time (sec)")
            plt.ylabel("Distance (cm)")

    # defaul dim=0 x axis 
    def MI_all_pair(self,dim=0):
        dms = range(-99, 100)
        time_lag = np.array(dms)*self.dt
        d = 0
        for i in range(0, 16):
            for j in range(i+1, 16):
                mi_11 = [_mi_quick(self.tra[1:, i, dim],self.tra[1:, i, dim], d) for d in dms]  # auto
                mi_22 = [_mi_quick(self.tra[1:, j, dim],self.tra[1:, j, dim], d) for d in dms]
                mi_12 = [_mi_quick(self.tra[1:, i, dim],self.tra[1:, j, dim], d) for d in dms]  # cross
                d += 1
                plt.figure(d)
                plt.plot(time_lag, mi_11)
                plt.plot(time_lag, mi_22)
                # too small, needed to magnification
                plt.plot(time_lag, np.array(mi_12)*3)

                plt.title("fish "+str(i+1)+" and "+str(j+1))
                plt.ylabel("Bit")
                plt.xlabel("Time lag (sec)")
                #plt.savefig("fish "+str(i+1)+" and "+str(j+1)+".jpg",dpi=300)
    
    #caculate two fishes MI
    def MI_two(self,i,j,dim=0,time=0):
        time = int(time/self.dt) if time else self.frame 
        dms = range(-99,100)
        time_lag = np.array(dms)*self.dt 
        mi_11 = [_mi_quick(self.tra[1:time, i, dim],self.tra[1:time, i, dim], d) for d in dms]  # auto
        mi_22 = [_mi_quick(self.tra[1:time, j, dim],self.tra[1:time, j, dim], d) for d in dms]
        mi_12 = [_mi_quick(self.tra[1:time, i, dim],self.tra[1:time, j, dim], d) for d in dms]  # cross
        plt.plot(time_lag, mi_11)
        plt.plot(time_lag, mi_22)
        # too small, needed to magnification
        plt.plot(time_lag, np.array(mi_12)*3)

        plt.title("fish "+str(i+1)+" and "+str(j+1))
        plt.ylabel("Bit")
        plt.xlabel("Time lag (sec)")    
    
    #vector and vector sum caculator
    def zbfvec(self):
        for i in range(0,self.noa):
            for j in range(0,self.frame-1):
                self.vec[j,i]=[self.tra[j+1,i,0]-self.tra[j,i,0],self.tra[j+1,i,1]-self.tra[j,i,1]]
                self.vec_s[j,i]=[self.tra[j+1,i,0]-self.tra[1,i,0],self.tra[j+1,i,1]-self.tra[1,i,1]]
                self.vec_rms[j,i]= np.sqrt((self.vec_s[j,i,0]**2+self.vec_s[j,i,0]**2))
    
    #vec plot 
    def zbfvecplt(self,typ="rms",dim=0):
        time=np.linspace(0,(self.frame-1)/30,self.frame-1)
        
        if (typ== "rms"):
            for i in range(0,self.noa):
                plt.figure(i)
                plt.title(str(i+1)+" fish vector rms")
                plt.xlabel("time (sec)")
                plt.ylabel("vector size")
                plt.plot(time,self.vec_rms[:,i])
                
        elif (typ=="sum"):
            for i in range(0,self.noa):
                plt.figure(i)
                plt.xlabel("time (sec)")
                plt.ylabel("vector size")
                plt.title(str(i+1)+" fish vector sum in y axis" if dim else str(i+1)+" fish vector sum in x axis")
                plt.plot(time,self.vec_s[:,i,dim])
            
        elif (typ=="vec"):
            for i in range(0,self.noa):
                plt.figure(i)
                plt.xlabel("time (sec)")
                plt.ylabel("vector size")
                plt.title(str(i+1)+" fish vector in y axis" if dim else str(i+1)+" fish vector in y axis")
                plt.plot(time,self.vec_s[:,i,dim])
    
    
    #two fishes distance 
    def zbfdis(self):
        #set a list of two different number 
        x = np.arange(0,self.noa)
        comb = combinations(x,2)
        y = list(comb)              
    
        time=np.linspace(0,self.frame*self.dt,self.frame)
        dis=np.empty((self.frame,len(y)))
        
        n=0
        for j in y:
            for i in range(0,self.frame):
                 dis[i-1,n]=np.sqrt((self.tra[i,j[0],0]-self.tra[i,j[1],0])**2+(self.tra[i,j[0],1]-self.tra[i,j[1],1])**2)/self.dtcm

            plt.figure(n)
            plt.title(str(j[0])+"-"+str(j[1]))
            plt.xlabel("time (sec)")
            plt.ylabel("distance (cm)")
            plt.plot(time,dis[:,n])
            n+=1




# interplate
def _uni(tra, frn, rear, j, k):
    n = rear - frn  # number of nan
    step = (tra[rear, j, k] - tra[frn-1, j, k])/max(n, 1)
    for i in range(1, n+1):
        tra[i+frn-1, j, k] = tra[frn-1, j, k] + step*i

# load data and interpolate in nan
def _traload(path, noa):
    tra = np.load(path, allow_pickle=True)  # load data
    tra = tra.item()
    tra = tra['trajectories']  # get data
    front = 0
    frame = len(tra)
    for j in range(0, noa):
        for k in range(0, 2):
            flag = False
            for i in range(1, frame):
                if (np.isnan(tra[i, j, k]) and flag == False):
                    flag = True
                    front = i
                elif ((not np.isnan(tra[i, j, k])) and flag == True):
                    flag = False
                    _uni(tra, front, i, j, k)
    return tra



# using numpy's histogram2d to calculate the mutual information between two sequences
def _mi_quick(a, b, d, bn=25):
    if d > 0:
        xy, _, _ = np.histogram2d(a[d:], b[:-d], bn)
    elif d < 0:
        xy, _, _ = np.histogram2d(a[:d], b[-d:], bn)
    else:
        xy, _, _ = np.histogram2d(a, b, bn)
    xy /= np.sum(xy)
    px = [np.array([max(x, 1e-100) for x in np.sum(xy, axis=0)])]
    py = np.transpose([[max(x, 1e-100) for x in np.sum(xy, axis=1)]])
    nxy = (xy/px)/py
    nxy[nxy == 0] = 1e-100
    return np.sum(xy*np.log2(nxy))
