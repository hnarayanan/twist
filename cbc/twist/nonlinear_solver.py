__author__ = "Marek Netusil"


from dolfin import *
from numpy import linalg


def default_solver_parameters():
    " Default parameters for the Newton solver if none are given "

    p = Parameters("solver_parameters")
    p.add("maximum_iterations", 50)
    p.add("absolute_tolerance", 1E-8)
    p.add("relative_tolerance", 1E-7)
    p.add("loading_number_of_steps", 1) # if >1 => incremental loading used
    # Relaxation parameter setup - subparameter of solver_parameters
    rel = Parameters("relaxation_parameter")
    rel.add("value", 1.0)       # starting value of the relaxation parameter
    rel.add("adaptive", True)   # should the solver use the adaptive approach?
    p.add(rel)

    return p
    


class AugmentedNewtonSolver():
    """ Newton solver with implemented damping and incremental loading """

    def __init__(self, F, u, a, bc, parameters = default_solver_parameters(), load_increment = None):
        """ Initialize the solver """

        # Get parameters
        self.parameters = parameters

        # Get problem data: F(u) == 0
        self.F = F          # equation
        self.u = u          # unknown
        self.bc = bc        # dirichlet boundary conditions
        self.a = a          # jacobian of F
        self.load_increment = load_increment    # incremental coefficient


    def solve(self):
        """ Solve the nonlinear system F(u) == 0 """

        relaxation = self.parameters["relaxation_parameter"]
        n = self.parameters["loading_number_of_steps"]
         

        # Choose the solver implementation - adaptive vs. fenics default

        # Solver with adaptive relaxation parameter
        # FIXME: Replace this homemade solver with an overloaded NonlinearVariationalSolver
        if relaxation["adaptive"]:          
            if not self.load_increment or n == 1:
                # Solve the problem with full loading
                self.adaptive()
            else:
                # Incremental loading implementation
                dtheta = 1.0/n              # define the increment
                # Solve the problem for the incremental coefficient ranging from 0.0 to 1.0
                for i in range(n+1):            
                    self.load_increment.assign(i*dtheta)
                    # FIXME: choose some log level for this output
                    print 'theta = ', float(self.load_increment)
                    self.adaptive()

        # Non-adaptive case - solve each load increment with the NonlinearVariationalSolver
        else:
            problem = NonlinearVariationalProblem(self.F, self.u, self.bc, self.a)
            solver = NonlinearVariationalSolver(problem)

            prm = solver.parameters
            prm["newton_solver"]["absolute_tolerance"] = self.parameters["absolute_tolerance"]
            prm["newton_solver"]["relative_tolerance"] = self.parameters["relative_tolerance"]
            prm["newton_solver"]["maximum_iterations"] = self.parameters["maximum_iterations"]
            prm["newton_solver"]["relaxation_parameter"] = relaxation["value"]

            if not self.load_increment or n == 1:
                solver.solve()
            else:
                # Incremental loading implementation
                dtheta = 1.0/n              # define the increment
                # Solve the problem for the incremental coefficient ranging from 0.0 to 1.0
                for i in range(n+1):            
                    self.load_increment.assign(i*dtheta)
                    # FIXME: choose some log level for this output
                    print 'theta = ', float(self.load_increment)
                    solver.solve()


    def adaptive(self):
        """ Simple homemade Newton solver """

        # Get parameters
        relaxation = self.parameters["relaxation_parameter"]
        absolute_tol = self.parameters["absolute_tolerance"]
        max_iter = self.parameters["maximum_iterations"]

        # Prepare the algebraic variables
        vector = self.u.function_space()
        u_temp = Function(vector)
        dx = Vector()
        A = Matrix()
        b = Vector()
        x = self.u.vector()

        # Initialize the controlling variables
        nIter = 0               # Number of performed iterations
        b = assemble(self.F)    # Residual
        for bc in self.bc:
            bc.apply(b, x)
        res_norm = b.norm('l2')

        # Log output - similar to the dolfin one
        print '  Newton iteration{0:2d}: r (abs) = {1:1.3e}'.format(nIter,res_norm)


        # Newton iterations
        while res_norm > absolute_tol and nIter < max_iter:
            if not dx.empty():
                dx.zero()
            nIter += 1
            u_temp.assign(self.u)   # Save the solution u_(k-1)
            # Set up and solve the algebraic system
            A = assemble(self.a)
            for bc in self.bc:
                bc.apply(b, x)
                bc.apply(A)
            solve(A, dx, b)

            # Set the relaxation parameter to 1.0
            omega = 1.0
            self.u.vector().axpy(-omega, dx)    # Compute u_k

            # Compute the absolute residual R_k
            b = assemble(self.F)
            for bc in self.bc:
                bc.apply(b, x)
            res_norm2 = b.norm('l2')
            
            # If residual R_k > R_(k+1), decrease the relaxation parameter by half
            while res_norm2 >= res_norm:
                omega = omega*0.5
                self.u.vector()[:] = u_temp.vector() - omega*dx
                b = assemble(self.F)
                for bc in self.bc:
                    bc.apply(b, x)
                res_norm2 = norm(b, 'l2')
            
            # Update the residual and print the log output
            res_norm = res_norm2
            print '  Newton iteration{0:2d}: r (abs) = {1:1.3e}    relaxation = {2:1.2e}'.format(nIter,res_norm,omega)

