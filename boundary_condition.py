class Dirichlet:
    def __init__(self, lower_bound, upper_bound, l_value, u_value):
        
        self.name = "Dirichlet"
        self.lower_bound =  lower_bound
        self.upper_bound =  upper_bound

        self.l_value = l_value
        self.u_value = u_value

class Neumann:
    def __init__(self, lower_bound, upper_bound, l_value, u_value):
        self.name = "Neumann"
        self.lower_bound =  lower_bound
        self.upper_bound =  upper_bound

        self.l_value = l_value
        self.u_value = u_value



