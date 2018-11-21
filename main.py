#import numpy as np
#import scipy.sparse as sp
#import scipy.sparse.linalg as spalg
from boundary_condition import Dirichlet, Neumann
from finite_element import FE
from finite_difference import FD
from spectrum import GLL_Spectrum
import matplotlib.pyplot as plt
import numpy as np

# Dirichlet_BD = Dirichlet(0.0, L, 0.0, 0.0)
# FD: fd = FD(c,L,pe,n,source=1.0)
# FE: fe = FE(diff_coeff,ad_coeff,f,BD,n)
# Specturm

L = 1.0
c = 1.0
BD = Dirichlet(0.0,L,0.0,0.0)
n = 100

def exact_solution(pe, x):
    return L/c*(x/L + (np.exp(-pe*(1-x/L)) - np.exp(-pe))/(np.exp(-pe)-1.0) )

def relative_error(exact,result):
    a = exact-result
    return np.linalg.norm((a[1:-1])/exact[1:-1], ord =np.inf)

def plot_Pe(pe):
    fd = FD(c,L,pe,n, source = 1.0)
    fe = FE(1.0, pe, pe*L/c, BD, 100)
    gl = GLL_Spectrum(1.0,pe,pe*L/c, BD,n)
    fd_result = fd.solve()
    fe_result = fe.solve()
    gl_result = gl.solve()
    plt.plot(fe.mesh,exact_solution(pe,fe.mesh),label="exact")
    plt.plot(fd.mesh,fd_result,label="FD")
    plt.plot(fe.mesh,fe_result,label="FE")
    plt.plot(gl.mesh,gl_result,label="GLL")

    plt.legend(loc="best")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.show()
    
    fd_error = relative_error(exact_solution(pe,fd.mesh),fd_result)
    fe_error = relative_error(exact_solution(pe,fe.mesh),fe_result)
    gl_error = relative_error(exact_solution(pe,gl.mesh),gl_result)

    return fd_error, fe_error, gl_error

def errors(pe,n):
    fd = FD(c,L,pe,n, source = 1.0)
    fe = FE(1.0, pe, pe*L/c, BD, n)
    gl = GLL_Spectrum(1.0,pe,pe*L/c, BD,n)
    fd_result = fd.solve()
    fe_result = fe.solve()
    gl_result = gl.solve()

    fd_error = relative_error(exact_solution(pe,fd.mesh),fd_result)
    fe_error = relative_error(exact_solution(pe,fe.mesh),fe_result)
    gl_error = relative_error(exact_solution(pe,gl.mesh),gl_result)
    del fd,fe,gl
    return fd_error, fe_error, gl_error


def multiple_pe_errors(n, pe_list):
    N = len(pe_list)
    fd_elist = np.zeros(N)
    fe_elist = np.zeros(N)
    gl_elist = np.zeros(N)

    for i in range(N):
        print str(pe_list[i])+" ... "
        fd_elist[i], fe_elist[i] , gl_elist[i] = errors(pe_list[i],n)
    return fd_elist, fe_elist, gl_elist

def problem_b():
    fd_error, fe_error, gl_error =plot_Pe(100.0)
    print 
    print "finite difference error : " + str(fd_error)
    print "finite element  error : " + str(fe_error)
    print "GLL spectrum error : " + str(gl_error)
    print

    fd_error, fe_error, gl_error =plot_Pe(1000.0)
    print 
    print "finite difference error : " + str(fd_error)
    print "finite element  error : " + str(fe_error)
    print "GLL spectrum error : " + str(gl_error)
    print

def problem_c():
    pe_list = np.array( [1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8] )
    plt.figure()
    fd_elist, fe_elist, gl_elist = multiple_pe_errors(100,pe_list)
    plt.plot(pe_list,fd_elist,'o-',label="FD, n=100")
    plt.plot(pe_list,fe_elist,'o-',label="FE, n=100")
    plt.plot(pe_list,gl_elist,'o-',label="GLL, n=100")
    fd_elist, fe_elist, gl_elist = multiple_pe_errors(101,pe_list)
    plt.plot(pe_list,fd_elist,'^-',label="FD, n=101")
    plt.plot(pe_list,fe_elist,'^-',label="FE, n=101")
    plt.plot(pe_list,gl_elist,'^-',label="GLL, n=101")
    
    plt.legend(loc="best")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()


#problem_c()
fd_error, fe_error, gl_error =plot_Pe(1.0e6)
print 
print "finite difference error : " + str(fd_error)
print "finite element  error : " + str(fe_error)
print "GLL spectrum error : " + str(gl_error)
print



