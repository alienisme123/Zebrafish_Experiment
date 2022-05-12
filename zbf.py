import numpy as np
import matplotlib.pyplot as plt


class zbf:
    def __init__(self, path, noa, dtcm):  # path/number of animals/dctm = digtal to cm
        self.noa = noa
        self.tra = traload(path, noa)
        self.frame = len(self.tra)
        self.dtcm = dtcm

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
    def oned_to_time(self, dim=0, dt=1/30):
        time = np.linspace(0, len(self.tra)*dt, len(self.tra))
        axis = "Y" if dim else "X"
        for i in range(0, self.noa):
            plt.figure(i+self.noa*dim)
            plt.plot(time, self.tra[:, i, dim]/self.dtcm)
            plt.title(axis+"-time "+str(i+1)+" fish")
            plt.xlabel("time (sec)")
            plt.ylabel("Distance (cm)")


# interplate
def uni(tra, frn, rear, j, k):
    n = rear - frn  # number of nan
    step = (tra[rear, j, k] - tra[frn-1, j, k])/max(n, 1)
    for i in range(1, n+1):
        tra[i+frn-1, j, k] = tra[frn-1, j, k] + step*i

# load data and interpolate in nan


def traload(path, noa):
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
                    uni(tra, front, i, j, k)
    return tra
