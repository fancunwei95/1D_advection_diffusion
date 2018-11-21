
class FE:
    def __init__(self, diff_coeff, ad_coeff , f, BD, n, grid_method="chebyshev", global_solve= True):
        
        global np, sp, spalg

        import numpy as np
        import scipy.sparse as sp
        import scipy.sparse.linalg as spalg
 
        if type(diff_coeff) == float or int:
            my_diff_coeff = lambda x : (x*x+1.0)/(x*x+1.0)*diff_coeff
        else:
            my_diff_coeff = diff_coeff
        if type(ad_coeff) == float or int:
            my_ad_coeff = lambda x: ad_coeff*(x*x+1.0)/(x*x+1.0)
        else:
            my_ad_coeff = ad_coeff
        if type(f) ==float or int:
            my_f = lambda x : f*(x*x+1.0)/(x*x+1.0)
        else:
            my_f = f
        
        self.diff_coeff = my_diff_coeff
        self.ad_coeff = my_ad_coeff
        self.f = my_f
        self.Boundary = BD
        self.n = n
        if grid_method.lower() not in ["chebyshev","uniform"]:
            raise Exception("gird_method "+str(grid_method)+" is not supported")
        self.grid_method = grid_method
        self.L = BD.upper_bound - BD.lower_bound
        self.global_solve = global_solve

    def get_grid(self):
        # the -1 to 1 grid
        n = self.n
        if self.grid_method.lower() == "uniform":
            self.grid = np.linspace(-1,1,n+2)
        if self.grid_method.lower() =="chebyshev":
            index = np.linspace(n+1,0,n+2)
            self.grid = np.cos(np.pi*index/(n+1.0))
        
    def convert(self,x):
        a = self.Boundary.lower_bound
        b = self.Boundary.upper_bound
        return (x+1.0)/2.0*(b-a) +a 
 
    def global_diff_matrix(self):
        
        grid = self.grid
        p = self.diff_coeff( self.convert(self.grid[:-1] + self.grid[1:])*0.5 )
        diag = np.zeros(self.n+2)
        diag[1:-1] = 1.0/(grid[1:-1]-grid[:-2])*p[:-1] + 1.0/(grid[2:] - grid[1:-1])*p[1:]
        diag[0] = 1.0/(grid[1] - grid[0])*p[0]
        diag[-1] = 1.0/(grid[-1] -grid[-2])*p[-1]
        off_diag = -1.0/(grid[1:]-grid[:-1])*p
        diff_matrix = sp.diags( [off_diag,diag,off_diag] ,[-1,0,1])
        
        return diff_matrix/self.L*2.0 

    def global_ad_matrix(self):
        # used the 0 th order gaussian quadrature
        
        c = self.ad_coeff( self.convert((self.grid[:-1] + self.grid[1:])*0.5) )
        adv_matrix = sp.diags([-0.5*c,0.5*c],[-1,1],shape=(self.n+2,self.n+2))
        
        return adv_matrix

    def Dirichlet_matrix(self):
        
        R_ = sp.diags([1],[1], shape=(self.n, self.n +1))
        zeros = sp.diags([0],[0],shape=(self.n, 1) )
        R = sp.hstack([R_,zeros])
        return R

    def Dirichlet_wrapper(self,matrix):

        R_csr = sp.csr_matrix(self.Dirichlet_matrix())
        M = sp.csr_matrix(matrix)

        return R_csr.dot(M.dot(R_csr.transpose()))

    def global_b_matrix(self):
        # use gaussian quadrature of zeros order in each linear interval
        f = self.f( self.convert( (self.grid[:-1] + self.grid[1:])*0.5) ) 
        b = np.zeros(self.n+2)
        b[1:-1] = 0.5*(self.grid[1:-1] - self.grid[:-2])*f[:-1]+0.5*(self.grid[2:]-self.grid[1:-1])*f[1:]
        b[0] = 0.5*(self.grid[1] - self.grid[0])*f[0]
        b[-1] = 0.5*(self.grid[-1] - self.grid[-2])*f[-1]
        return b*self.L*0.5

    def global_Ab(self):
        self.get_grid()
        A_matrix = self.global_diff_matrix() + self.global_ad_matrix()
        b_matrix = self.global_b_matrix()
        return A_matrix, b_matrix

    def local_diff_matrix(self):

        pass

    def local_ad_matrix(self):

        pass

    def local_b_matrix(self):
        pass

    def assembly_Ab(self):
        pass

    def solve(self):
        result = np.zeros(self.n+2)
        if self.global_solve:
            A,b = self.global_Ab()
        else:
            A,b = self.assembly_Ab()
        if self.Boundary.name == "Dirichlet":
            R = sp.csr_matrix(self.Dirichlet_matrix())
            A = R.dot(sp.csr_matrix(A).dot(R.transpose()) )
            b = R.dot(b)
            result = spalg.spsolve(A,b)
            result = sp.csr_matrix(self.Dirichlet_matrix()).transpose().dot(result)
        self.mesh = self.convert(self.grid)
        return result


if __name__=="__main__":
    from boundary_condition import Dirichlet
    import matplotlib.pyplot as plt
    c = 1.0
    L = 1.0
    pe = 100.0
    f = pe*L/c
    n = 101
    Dirichlet_BD = Dirichlet(0.0,L,0.0,0.0)
    fe = FE(1.0,pe,f,Dirichlet_BD,n)

    result = fe.solve()

    plt.plot(fe.mesh,result)
    plt.show()

