import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

# Промежуток времени, на котором рассчитывается симуляция, и его разбивка на кадры
tfin = 20
frameCount = tfin * 70
Steps = frameCount * 2
t = np.linspace(0, tfin, Steps)

global slope, m1, m2, coeffC, coeffK, g, lenStick
g = 9.81; slope = 0.01; m1 = 2; m2 = 0.25; lenStick = 0.2;
coeffC = 0.1
coeffK = 0.05 # первый вариант значения
coeffK = 0.03 # второй вариант значения


# эта функция выражает вторые производные обобщенных координат методом Крамера
def SystemOfEquations(y, time):
    yt = np.zeros_like(y)
    yt[0] = y[2]
    yt[1] = y[3]

    a11 = 2*(m1+m2)
    a12 = -m2*lenStick*np.cos(y[1]+slope)
    a21 = -m2/2*lenStick*np.cos(y[1]+slope)
    a22 = m2/3*lenStick**2

    b1 = 2*(m1 + m2)*g*np.sin(slope) - m2*lenStick*y[3]**2*np.sin(y[1]+slope)
    b2 = m2/2*lenStick*g*np.sin(y[1]) - coeffK*y[3] - coeffC*y[1]
    det = (a11*a22 - a21*a12)

    yt[2] = (b1*a22 - b2*a12)/det
    yt[3] = (a11*b2 - a21*b1)/det

    return yt


S0 = 0; Phi0 = -0.5236; DS0 = 0; DPhi0 = 4; y0 = [S0, Phi0, DS0, DPhi0]  # Начальные условия
Y = odeint(SystemOfEquations, y0, t)
s = Y[:, 0]
phi = Y[:, 1]
Ds = Y[:, 2]
Dphi = Y[:, 3]
DDs = [SystemOfEquations(y, t)[2] for y, t in zip(Y, t)]
DDphi = [SystemOfEquations(y, t)[3] for y, t in zip(Y, t)]

Nreac = g*(m1+m2)*np.cos(slope) - m2/2*lenStick*(DDphi*np.sin(phi+slope) + Dphi**2*np.cos(phi+slope))

# весь дальнейший код - рисовка графики.

# Следующие 3 блока кода - рисовка графиков s(t), phi(t) и N(t)
fig_graphs = plt.figure(figsize=[15, 15])
ax_graphs = fig_graphs.add_subplot(3, 1, 1)
ax_graphs.plot(t, s, color="red")
ax_graphs.set_title("s(t)")
ax_graphs.set(xlim=[0, tfin])
ax_graphs.grid("True")

ax_graphs = fig_graphs.add_subplot(3, 1, 2)
ax_graphs.plot(t, phi, color="blue")
ax_graphs.set_title("phi(t)")
ax_graphs.set(xlim=[0, tfin])
ax_graphs.grid("True")

ax_graphs = fig_graphs.add_subplot(3, 1, 3)
ax_graphs.plot(t, Nreac, color="black")
ax_graphs.set_title("N(t)")
ax_graphs.set(xlim=[0, tfin])
ax_graphs.grid("True")

# Определение длин сторон призмы
lenHypot = 0.2
lenUp = lenHypot * np.cos(slope)
lenRight = lenHypot * np.sin(slope)

# Определение спиральной пружины
CoilNum = 2
R1 = lenUp / 20
R2 = lenUp / 6
theta = np.linspace(0, CoilNum * 6.28 - phi[0], 100)
XSpiralSpr = (R1 + theta * (R2 - R1)/theta[-1]) * np.sin(theta)
YSpiralSpr = (R1 + theta * (R2 - R1)/theta[-1]) * np.cos(theta)

# Зарисовка поверхностей
xGr1 = np.linspace(0, 5 * lenHypot, 10)
xGr2 = np.linspace(0, lenHypot / 3, 10)
DiagLine1 = -np.tan(slope) * xGr1
DiagLine2 = xGr2 / np.tan(slope)

# Описание движения графики призмы и точки крепления стержня
PrismX = np.array([-lenUp * 0.9, lenUp * 0.1, lenUp * 0.1, -lenUp * 0.9])
PrismY = np.array([lenRight / 2, lenRight / 2, -lenRight / 2, lenRight / 2])
XO = 0.9 * lenUp + s*np.cos(slope)
YO = DiagLine1[0] - lenRight / 2 - s*np.sin(slope)

# Зарисовка кучи элементов.
fig = plt.figure(figsize=[15, 15])
ax = fig.add_subplot(1, 1, 1)
ax.grid("True")
ax.axis('equal')
ax.set(xlim=[-1, xGr1[-1] + 2], ylim=[-1, YO[0] + lenStick])
ax.plot([0, xGr1[-1]], [DiagLine1[-1], DiagLine1[-1]], color='k')
ax.plot(xGr1, DiagLine1, color='k')
ax.plot(xGr2, DiagLine2, '--', color='k')
ax.plot(xGr1 * 2, DiagLine1 * 2, '--', color='k')
PointO = ax.plot(XO[0], YO[0], marker='o')[0]  # Точка, в которой крепится стержень
Line = ax.plot([XO[0], XO[0] - lenStick * np.sin(phi[0])], [YO[0], YO[0] + lenStick * np.cos(phi[0])])[0]  # Стержень
DrawnPrism = ax.plot(XO[0] + PrismX, YO[0] + PrismY)[0]
DrawnSpiralSpring = ax.plot(XSpiralSpr + XO[0], YSpiralSpr + YO[0])[0]

def anim(i):
    PointO.set_data(XO[i], YO[i])
    DrawnPrism.set_data(XO[i] + PrismX, YO[i] + PrismY)
    Line.set_data([XO[i], XO[i] - lenStick * np.sin(phi[i])], [YO[i], YO[i] + lenStick * np.cos(phi[i])])
    theta = np.linspace(0, CoilNum * 6.28 - phi[i], 100)
    XSpiralSpr = (R1 + theta * (R2 - R1) / theta[-1]) * np.sin(theta)
    YSpiralSpr = (R1 + theta * (R2 - R1) / theta[-1]) * np.cos(theta)
    DrawnSpiralSpring.set_data(XSpiralSpr + XO[i], YSpiralSpr + YO[i])
    return [PointO, DrawnPrism, Line, DrawnSpiralSpring]


film = FuncAnimation(fig, anim, interval=3, frames=frameCount)
plt.show()
