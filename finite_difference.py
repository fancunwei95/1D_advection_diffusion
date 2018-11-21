#import numpy as np
#import scipy.sparse as sp
#import scipy.sparse.linalg as spalg

class FD:
    def __init__(self, c , L, pe, n, source=0.0, source_func=None):

        global np, sp, spalg

        import numpy as np
        import scipy.sparse as sp
        import scipy.sparse.linalg as spalg


        self.c = c
        self.L = L 
        self.pe = pe
        self.n = n
        self.source_num = source
        self.source_func = source_func 
    
    def mesh(self):
        n = self.n
        L = self.L
        self.mesh = np.linspace(0,L,n+2)
        self.dx = self.mesh[1] - self.mesh[0]
        return 

    def source(self):
        if self.source_func == None:
            self.source = np.ones(len(self.mesh))*self.source_num*self.pe*self.L/self.c
        else:
            self.source = self.source_func(self.mesh)*self.pe*self.L/self.c
        return

    def A_matrix(self):
        diffu_matrix = 1.0/self.dx/self.dx*sp.diags([-1.0,2.0,-1.0],[-1,0,1], shape=(self.n,self.n))
        adv_matrix = 1.0/2.0/self.dx*sp.diags([-1.0,1.0],[-1,1],shape=(self.n,self.n))
        A_matrix = diffu_matrix + self.pe*adv_matrix
        return A_matrix
    
    def initialize(self):
        self.mesh()
        self.source()

    def solve(self):
        self.initialize()
        A= self.A_matrix()
        b= self.source[1:-1]
        raw_result = spalg.spsolve(A,b)
        result = np.zeros(raw_result.shape[0]+2)
        result[1:-1] = raw_result
        return result



if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    c = 1.0
    L = 1.0
    pe = 100.0
    n = 100
    source = 1.0
    fd = FD(c,L,pe,n,source= 1.0)
    result = fd.solve()
    plt.plot(fd.mesh, result)
    plt.show()
