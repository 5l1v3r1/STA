# -*- coding: utf-8 -*-

import numpy as np
import sympy as sp
from qutip import *
from scipy import interpolate
from scipy import integrate
import matplotlib.pyplot as plt

class OQA:

    ## The constructor. This constructor generates Hamiltonian H=A(s(t))X+B(s(t))Z+C(s(t))X2
    # @param X  qutip Hermitian operator
    # @param Z  qutip Hermitian operator
    # @param T  total annealing time
    # @param X2 qutip Hermitian operator (additional)
    # @param division1 the schedules A(s), B(s), and C(s) (0<=s<=1) consist of division1 points
    # @param division2 the schedule s(t) consists of division2 points 
    def __init__(self, X, Z, T, X2=0, division1=100, division2=200):
        self.X = X # qutip Hermitian operator
        self.Z = Z # qutip Hermitian operator
        if X2 is 0:
            self.X2 = 0*self.X # qutip Hermitian operator (additional)
        else:
            self.X2 = X2
        self.T = T # total annealing time
        #num: the length of slist

        #assume that the form of the Hamiltonian is AX + BZ + CX2

        #initialize all lists
        self.slist = np.linspace(0,1,division1) # (0,1)
        self.Alist = 1-self.slist # A(s) as a function of s #A(0)=1, A(1)=0
        self.Blist = self.slist # B(s) as a function of s #B(0)=0, B(1)=1
        self.Clist = -self.slist*(self.slist-1) # C(s) as a function of s #C(0)=C(1)=0
        # Each of the length of Alist, Blist, and slist must be the same
        # construct (s,t) pair
        self.stlist_s = np.linspace(0,1,division2) # (0,1)
        self.stlist_t = self.T*self.stlist_s # (0,T)
        # Each of the length of s and t must be the same

    
    ## A(s)
    # @param s the schedule (0<=s<=1)
    def A(self, s):
        f = interpolate.interp1d(self.slist, self.Alist, kind="cubic", fill_value="extrapolate")
        return float(f(s))

    ## B(s)
    # @param s the schedule (0<=s<=1)
    def B(self, s):
        f = interpolate.interp1d(self.slist, self.Blist, kind="cubic", fill_value="extrapolate")
        return float(f(s))

    ## C(s)
    # @param s the schedule (0<=s<=1)
    def C(self, s):
        f = interpolate.interp1d(self.slist, self.Clist, kind="cubic", fill_value="extrapolate")
        return float(f(s))

    ## the differenciation of A(s)
    # @param s the schedule (0<=s<=1)
    def dAds(self, s, ds=0.0001):
        return float((self.A(s-2*ds)-8*self.A(s-ds)+8*self.A(s+ds)-self.A(s+2*ds))/(12*ds))

    ## the differenciation of B(s)
    # @param s the schedule (0<=s<=1)
    def dBds(self, s, ds=0.0001):
        return float((self.B(s-2*ds)-8*self.B(s-ds)+8*self.B(s+ds)-self.B(s+2*ds))/(12*ds))

    ## the differenciation of C(s)
    # @param s the schedule (0<=s<=1)
    def dCds(self, s, ds=0.0001):
        return float((self.C(s-2*ds)-8*self.C(s-ds)+8*self.C(s+ds)-self.C(s+2*ds))/(12*ds))

    ## s(t)
    # @param t the time (0<=t<=T)
    def s(self, t):
        f = interpolate.interp1d(self.stlist_t, self.stlist_s, kind="cubic", fill_value="extrapolate")
        return float(f(t))

    ## the differenciation of s(t)
    # @param t the time (0<=t<=T)
    def dsdt(self, t, dt=0.0001):
        return float((self.s(t-2*dt)-8*self.s(t-dt)+8*self.s(t+dt)-self.s(t+2*dt))/(12*dt))

    ## obtain Hamiltonian list (X initial op, Z final op, X2 intermediate op, A, B, C interpolated function as a function of t)
    def Hlist(self):
        return [[self.X, lambda t, args: self.A(self.s(t))],[self.Z, lambda t, args: self.B(self.s(t))],[self.X2, lambda t, args: self.C(self.s(t))]]

    ## H(t)
    # @param t the time (0<=t<=T)
    def H(self, t):
        return self.A(self.s(t))*self.X+self.B(self.s(t))*self.Z+self.C(self.s(t))*self.X2

    ## prepare the initial state
    def init_state(self, sparse=False):
        eigvals, eigvecs = self.H(0).eigenstates(sparse, eigvals=1)
        return eigvecs[0]

    ## return the variable stlist_t
    def tlist(self):
        return self.stlist_t

    ## generate counter diabatic term numerically
    def Hcd(self, t, dt=0.0001, ds=0.0001, sparse=False):
        eigvals_t, eigvecs_t = self.H(t).eigenstates(sparse)
        dHdt = self.dsdt(t, dt)*(self.dAds(self.s(t), ds)*self.X+self.dBds(self.s(t), ds)*self.Z+self.dCds(self.s(t), ds)*self.X2)
        ret = 0
        for m in range(len(eigvals_t)):
            for n in range(len(eigvals_t)):
                if m != n:
                    ret = ret + 1j*(1.0/(eigvals_t[n]-eigvals_t[m]))*eigvecs_t[m]*eigvecs_t[m].dag()*dHdt*eigvecs_t[n]*eigvecs_t[n].dag()

        return ret


    # candidate: cost function
    def cost_intTrHcd2(self, dt=0.0001, ds=0.0001, sparse=False):
        #define cost function
        f = np.vectorize(lambda t: ((self.Hcd(t, dt, ds, sparse))**2).tr().real)
        cost = f(self.tlist())
        itp = lambda t: float(interpolate.interp1d(self.tlist(), cost, kind="cubic", fill_value="extrapolate")(t))
        return integrate.quad(itp, 0, self.T)
        
    # candidate: cost function
    def cost_intGSHcd2GS(self, dt=0.0001, ds=0.0001, sparse=False):
        f = np.vectorize(lambda t: expect((self.Hcd(t, dt, ds, sparse))**2, self.H(t).eigenstates(sparse, eigvals=1)[1][0]).real)
        cost = f(self.tlist())
        itp = lambda t: float(interpolate.interp1d(self.tlist(), cost, kind="cubic", fill_value="extrapolate")(t))
        return integrate.quad(itp, 0, self.T)

    # candidate: cost function
    def cost_intsqrtTrHcd2(self, dt=0.0001, ds=0.0001, sparse=False):
        f = np.vectorize(lambda t: np.sqrt(((self.Hcd(t, dt, ds, sparse))**2).tr().real))
        cost = f(self.tlist())
        itp = lambda t: float(interpolate.interp1d(self.tlist(), cost, kind="cubic", fill_value="extrapolate")(t))
        return integrate.quad(itp, 0, self.T)

    # candidate: cost function
    def cost_intsqrtGSHcd2GS(self, dt=0.0001, ds=0.0001, sparse=False):
        f = np.vectorize(lambda t: np.sqrt(expect((self.Hcd(t, dt, ds, sparse))**2, self.H(t).eigenstates(sparse, eigvals=1)[1][0]).real))
        cost = f(self.tlist())
        itp = lambda t: float(interpolate.interp1d(self.tlist(), cost, kind="cubic", fill_value="extrapolate")(t))
        return integrate.quad(itp, 0, self.T)



if __name__ == '__main__':
    #test
    # T represents the total annealing time.
    for T in [1,2,4]: 
        print("T={0} calculation".format(T))
        # spin 1 system 
        Z = -jmat(1, 'z')**3
        X = -jmat(1, 'x')  
        #magnetization operator
        M = jmat(1, 'z')
        obj = OQA(X, Z, T)

        # 2 spin system with small magnetic field
        #Z = tensor(sigmaz(), sigmaz())+0.5*tensor(sigmaz(), qeye(2))
        #X = tensor(sigmax(), qeye(2))+tensor(qeye(2), sigmax())
        #X2 = tensor(sigmax(), sigmax())
        #magnetization
        #M = tensor(sigmaz(), qeye(2))+tensor(qeye(2), sigmaz())
        #obj = OQA(X, Z, T, X2)

        #sample schedule:
        #A(s)=1-s
        obj.Alist = 1-obj.slist
        #B(s)=s
        obj.Blist = obj.slist
        #C(s)=0
        obj.Clist = obj.slist*0
        #s(t)=t/T

        # calculate the schrodinger dynamics
        result = mesolve(obj.Hlist(), obj.init_state(), obj.tlist(), [], [M], progress_bar=True)
        # generate the answer (the ground state at each time t)
        answer = []
        for t in obj.tlist():
            eigvals, eigvecs = obj.H(t).eigenstates(eigvals=1)
            answer.append(expect(M, eigvecs[0]))

        #calculate the magnetization 
        print("m: {0}".format(result.expect[0][-1]))

        #calculate each cost function
        print("intTrHcd2,err: {0}".format(obj.cost_intTrHcd2()))
        print("intGSHcd2GS,err: {0}".format(obj.cost_intGSHcd2GS()))
        print("intsqrtTrHcd2,err: {0}".format(obj.cost_intsqrtTrHcd2()))
        print("intsqrtGSHcd2GS,err: {0}".format(obj.cost_intsqrtGSHcd2GS()))

        #plot the magnetization of the state derived from schrodinger dynamics and the answer
        plt.plot(list(obj.tlist()), list(result.expect[0]), '.', list(obj.tlist()), list(answer))
        plt.title("T={0}".format(T))
        plt.show()

        # with CD term
        print("T={0} calculation with STA".format(T))
        result = mesolve(lambda t, args: obj.H(t)+obj.Hcd(t), obj.init_state(), obj.tlist(), [], [M], progress_bar=True)
        #plot the magnetization of the state derived from schrodinger dynamics and the answer
        plt.plot(list(obj.tlist()), list(result.expect[0]), '.', list(obj.tlist()), list(answer))
        plt.title("T={0}".format(T))
        plt.show()


