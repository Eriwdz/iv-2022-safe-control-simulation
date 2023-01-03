import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

X = np.genfromtxt("X_data.csv", delimiter = ',')

m = 1500.00

lf = 1.070

lr = 1.605

l = lf + lr

B = 1.517/2

Iz = 2600.00

Jw = 5.00

R = 0.316

b0 = 1.25

b1 = 5

T = 1.25

g = 9.81

Fz = m*g

c1, c2, c3 = 0.86, 33.82, 0.35

zeta = 0.85

xi = c1*Fz

P1 = 40

P2 = 0.01

vTire = np.zeros(4)
FTire = np.zeros([2, 2, 2])

q1 = np.array([-1, 1, -1, 1])
q2 = np.array([-1, 1, 1, -1])
q3 = np.array([1, 1, 0, 0])

epi = np.ones(4)*0.001

Ts = 1e-3

h_grad = lambda X: fprime(X, h, 1e-5)

def mu(s):
    return c1*(1 - np.exp(-c2*s)) - c3*s

def smoothMax(x1, x2):
    return 1/2*(x1 + x2 + np.sqrt(np.square(x2 - x1) + P2))

def smoothMin(x1, x2):
    x1, x2 = -x1, -x2
    return -1/2*(x1 + x2 + np.sqrt(np.square(x2 - x1) + P2))

def tireForces(x):
    F_test = np.zeros([4, 2])
    beta = lr/l*tan(x[10])
    vCOG = norm(x[3:5])*np.ones(4)
    vR = x[6:10]*R
    vTire = vCOG + q1*(B*np.cos(beta) - q2*lf*np.sin(beta))*x[5]
    alphaF = -beta + x[10] - lf*x[5]/vCOG[0]
    alphaR = -beta + lr*x[5]/vCOG[0]
    Sl = (vR*(q3*np.cos(alphaF) + (1 - q3)*np.cos(alphaR)) - vTire)/smoothMax(smoothMax(vR*(q3*np.cos(alphaF) + (1 - q3)*np.cos(alphaR)), vTire), epi)
    Ss = (q3*tan(alphaF) + (1 - q3)*tan(alphaR))*expit(P1*Sl) + (q3*np.sin(alphaF) + (1 - q3)*np.sin(alphaR))*vR/vTire*expit(-P1*Sl)
    Sr = norm(np.vstack((Sl, Ss)), axis = 0)
    Mu = mu(Sr)
    FL = Mu/Sr*Fz*Sl
    FS = Mu/Sr*Fz*Ss
    FTire[0, 0, :2] = (FL[:2]*np.cos(alphaF) + FS[:2]*np.sin(alphaF))*np.cos(x[10]) - (FS[:2]*np.cos(alphaF) - FL[:2]*np.sin(alphaF))*np.sin(x[10]) 
    FTire[1, 0, :2] = (FL[:2]*np.cos(alphaF) + FS[:2]*np.sin(alphaF))*np.sin(x[10]) + (FS[:2]*np.cos(alphaF) - FL[:2]*np.sin(alphaF))*np.cos(x[10])
    FTire[0, 1, :2] = FL[2:]*np.cos(alphaR) + FS[2:]*np.sin(alphaR)
    FTire[1, 1, :2] = -FL[2:]*np.sin(alphaR) + FS[2:]*np.cos(alphaR)
    F_test[0, :2] = FTire[0, 0, :2]
    F_test[1, :2] = FTire[1, 0, :2]
    F_test[2, :2] = FTire[0, 1, :2]
    F_test[3, :2] = FTire[1, 1, :2]
    return FTire, Sr

def h(F):
  Ffl = F[:, 0]
  Ffr = F[:, 1]
  Frl = F[:, 2]
  Frr = F[:, 3]
  hfl = 1 - (Ffl/(zeta*xi))**2
  hfr = 1 - (Ffr/(zeta*xi))**2
  hrl = 1 - (Frl/(zeta*xi))**2
  hrr = 1 - (Frr/(zeta*xi))**2
  return smoothMin(smoothMin(hfl, hfr), smoothMin(hrl, hrr))


X_data = np.genfromtxt("X_data.csv", delimiter = ',')

N = np.shape(X_data)[1]

t = np.arange(N)*Ts



plt.figure()

plt.plot(X_data[1], X_data[0], lw = 3.8, linestyle = '-', c = 'tab:orange')

plt.legend(fontsize = 28)

plt.xticks(fontsize = 30)

plt.yticks(fontsize = 30)

plt.xlabel('$X$(m)', fontsize = 35)

plt.ylabel('$Y$(m)', fontsize = 35)

plt.title('Vehicle Trajectory', fontsize = 35)

plt.xlim(7.5, 117)

plt.ylim(62, 90)

plt.grid()

plt.show()