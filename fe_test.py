import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spalg

class FE:
    def __init__(self, c, L, pe, n, source=0.0, source_func=None, 
        boundary_condition="Dirichlet", grid_method="c", symmetrized = False):
        # this class only solve diffusion equation with constant coefficients 
        # c is the advection velocity and L is the simulation region length
        # pe is the peclet number and n is the number of grids except boundaries
        # the source can be functions or constant 
        # symmetrized method only support constant source  
        # non symmetrized method support source function
        
        self.c = c
        self.L = L
        self.pe = pe
        self.n = n
        self.source_num = source
        self.source_func = source_func
        self.grid_method = grid_method
        self.symmetrized = symmetrized                 
        self.boundary_condition = boundary_condition
    
    def get_grid(self):
        n = self.n 
        if self.grid_method=="c":
            #index = np.linspace(0,n+1,n+2)
            index = np.linspace(n+1,0,n+2)
            grid = 0.5* ( 1.0 + np.cos(index*np.pi/(n+1.0)))
        self.grid = grid

    def change_grid_number(self,n):
        self.n = n

    def change_grid_method(self,grid_method):
        self.grid_method = grid_method
    
    def change_pe(self,pe):
        self.pe = pe
    
    def change_symmetry(self,symmetrized):
        self.symmetrized = symmetrized

    def A_b_matrix(self):
        if self.grid_method == "c": 
            if self.symmetrized :
                A =self.sym_Cheby_A_matrix()
                b =self.sym_Cheby_b_matrix()
            else:
                A =self.Cheby_A_matrix()
                b =self.Cheby_b_matrix()
        return A,b


    def sym_Cheby_A_matrix(self):
        grid = self.grid
        diag = np.zeros(self.n+2)
        diag[1:-1] = 1.0/(grid[1:-1]-grid[:-2]) + 1.0/(grid[2:] - grid[1:-1])
        diag[0] = 1.0/(grid[1] - grid[0])
        diag[-1] = 1.0/(grid[-1] -grid[-2])
        off_diag = -1.0/(grid[1:]-grid[:-1])
        
        adv_diag = np.zeros(self.n+2)
        adv_diag[1:-1] = 1.0/3.0*(grid[2:] - grid[:-2])
        adv_diag[0] = 1.0/3.0*(grid[1] - grid[0])
        adv_diag[-1] = 1.0/3.0*(grid[-1] - grid[-2])
        adv_off_diag = 1.0/6.0*(grid[1:]-grid[:-1])

        if self.boundary_condition == "Dirichlet":
            diag = diag[1:-1]
            off_diag = off_diag[1:-1]
            adv_diag = adv_diag[1:-1]
            adv_off_diag = adv_off_diag[1:-1]
            D_matrix = sp.diags([off_diag,diag,off_diag],[-1,0,1])
            adv_matrix = sp.diags([adv_off_diag,adv_diag,adv_off_diag],[-1,0,1])
            A_matrix= D_matrix + self.pe*self.pe/4.0*adv_matrix
        
        return A_matrix

    def sym_Cheby_b_matrix(self):
        grid = self.grid
        b_matrix=  np.zeros(self.n+2)
        part1 = np.exp(-self.pe/2.0*grid[2:])*(grid[1:-1] - grid[:-2])
        part2 = np.exp(-self.pe/2.0*grid[1:-1])*(grid[:-2] - grid[2:])
        part3 = np.exp(-self.pe/2.0*grid[:-2])*(grid[2:] - grid[1:-1])

        upper = 4.0*(part1+part2+part3)/self.pe/self.pe
        lower = (grid[:-2] - grid[1:-1])*(grid[1:-1]-grid[2:])
        b_matrix[1:-1] = upper/lower

        b_matrix[0] =  -4.0*np.exp(-0.5*self.pe* grid[0]) 
        b_matrix[0] += np.exp(-0.5*self.pe* grid[1])*(4-2*self.pe*(grid[0]-grid[1]))
        b_matrix[0] /= (grid[0] - grid[1])/self.pe/self.pe

        b_matrix[-1] = -4.0*np.exp(-0.5*self.pe*grid[-1]) 
        b_matrix[-1] += +np.exp(-0.5*self.pe*grid[-2])*(4-2*self.pe*(grid[-1]-grid[-2]))
        b_matrix[-1] /= (grid[-2]-grid[-1])/self.pe/self.pe
        
        b_matrix *=self.pe*self.L/self.c*self.source_num
        if self.boundary_condition == "Dirichlet":
            b_matrix= b_matrix[1:-1]
        return b_matrix

    def Cheby_A_matrix(self):
        grid = self.grid
        diag = np.zeros(self.n+2)
        diag[1:-1] = 1.0/(grid[1:-1]-grid[:-2]) + 1.0/(grid[2:] - grid[1:-1])
        diag[0] = 1.0/(grid[1] - grid[0])
        diag[-1] = 1.0/(grid[-1] -grid[-2])
        off_diag = -1.0/(grid[1:]-grid[:-1])

        if self.boundary_condition == "Dirichlet":
            diag = diag[1:-1]
            off_diag = off_diag[1:-1]
            D_matrix = sp.diags([off_diag,diag,off_diag],[-1,0,1])
            adv_matrix = sp.diags([-0.5,0.5],[-1,1],shape=(self.n,self.n))
            A_matrix = D_matrix + self.pe*adv_matrix
        
        return A_matrix

    def Cheby_b_matrix(self):
        b_matrix = np.zeros(self.n+2)
        b_matrix[1:-1] = self.grid[2:]-self.grid[:-2]
        b_matrix[0] = self.grid[1] -self.grid[0]
        b_matrix[-1] = self.grid[-1] - self.grid[-2]
        b_matrix *= 0.5*self.pe*self.L/self.c
        if self.boundary_condition == "Dirichlet":
            b_matrix = b_matrix[1:-1]
        
        return b_matrix

    def solve(self):
        self.get_grid()
        A,b = self.A_b_matrix()
        raw_result = spalg.spsolve(A,b)
        if self.boundary_condition =="Dirichlet":
            result = np.zeros(raw_result.shape[0]+2)
            result[1:-1] = raw_result
        else:
            result = raw_result
        
        if self.symmetrized:
            result *= np.exp(self.pe/2.0*self.grid)
        self.mesh = self.grid*self.L
        return result
    

if __name__=="__main__":
    import matplotlib.pyplot as plt
    c= 1.0
    L = 1.0
    pe = 100.0
    n =100
    source = 1.0
    fe_sym = FE(c,L,pe,n,source=1.0,symmetrized=True)
    fe = FE(c,L,pe,n,source=1.0,symmetrized=False)
    result = fe.solve()
    result_sym = fe_sym.solve()

    plt.plot(fe.grid,result,label="direct")
    plt.plot(fe_sym.grid,result_sym,label="symmetrized")
    plt.legend(loc="best")
    plt.show()
