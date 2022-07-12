import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
from scipy.integrate import odeint

#yt - по факту массив производных от функций оригинальных
#y - массив обобщенных координат. 1 - 2 - обычные функции, 3 - 4 - скорости

global m, M, L, g, c1, c2
m = 1; M = 5; L = 1; g = 9.81; c1 = 20; c2 = 20;
def SystemOfEquations(y, t):
    yt = np.array([1, 4])
    yt = np.zeros_like(y)
    yt[0] = y[2]
    yt[1] = y[3]

    a11 = m + M
    a12 = m*L*np.cos(y[1])
    a21 = m*L*np.cos(y[1])
    a22 = m*L**2

    b1 = -c1*y[0] + m*L*(y[3])**2*np.sin(y[1])
    b2 = -c2*y[1] + m*g*L*np.sin(y[1])

    yt[2] = (b1*a22 - b2*a12) / (a11 * a12 - a21 * a22)
    yt[3] = (a11*b2 - a12*b1) / (a11 * a12 - a21 * a22)

    return yt


X0 = 1; Phi0 = 1; DX0 = 0; DPhi0 = 0; y0 = [X0, Phi0, DX0, DPhi0]
Tfin = 10
FrameCount = 1001
t = np.linspace(0, Tfin, FrameCount)

Y = odeint(SystemOfEquations, y0, t)

x = Y[:,0]
phi = Y[:,1]

DDx = [SystemOfEquations(y, t)[2] for y,t in zip(Y, t)] #zip - совмещает 2 массива, чтобы читать парой пару массивов
DDphi = [SystemOfEquations(y, t)[3] for y,t in zip(Y, t)]
#вторые производные для исходных функций