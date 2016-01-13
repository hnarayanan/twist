from dolfin import *
from numpy import linalg


def default_solver_parameters():
    p = Parameters("solver_parameters")
    rel = Parameters("relaxation_parameter")
    rel.add("value", 1.0)
    rel.add("adaptive", True)
    p.add(rel)
    p.add("loading_number_of_steps", 5)
    p.add("maximum_iterations", 50)
    p.add("absolute_tolerance", 1E-8)
    p.add("relative_tolerance", 1E-7)

    return p
    


class AugmentedNewtonSolver():
    """ Newton solver with implemented damping and incremental loading """

    def __init__(self, F, u, du, bc, parameters = default_solver_parameters(), load_increment = None):
        self.parameters = parameters
        self.F = F
        self.u = u
        self.du = du
        self.bc = bc
        self.a = derivative(F, u, du)
        self.load_increment = load_increment

    def solve(self):
        F = self.F
        a = self.a
        u = self.u
        bc = self.bc
        relaxation = self.parameters["relaxation_parameter"]
        n = self.parameters["loading_number_of_steps"]
        

        if relaxation["adaptive"]:
            if self.load_increment == None:
                self.adaptive(F, u, bc, a)
            else:
                self.load_increment.assign(0.0)
                bcu_du = homogenize(bc)
                dtheta = 1.0/n
                
                for i in range(n+1):
                    print 'theta = ', float(self.load_increment)
                    self.adaptive(F, u, bcu_du, a)
                    self.load_increment.assign((i+1)*dtheta)
        else:
            problem = NonlinearVariationalProblem(F, u, bc, a)
            solver = NonlinearVariationalSolver(problem)

            prm = solver.parameters
            prm["newton_solver"]["absolute_tolerance"] = self.parameters["absolute_tolerance"]
            prm["newton_solver"]["relative_tolerance"] = self.parameters["relative_tolerance"]
            prm["newton_solver"]["maximum_iterations"] = self.parameters["maximum_iterations"]
            prm["newton_solver"]["relaxation_parameter"] = relaxation["value"]

            if self.load_increment == None:
                solver.solve()
            else:
                self.load_increment.assign(0.0)

                dtheta = 1.0/n
                
                for i in range(n+1):
                    print 'theta = ', float(self.load_increment)
                    solver.solve()
                    self.load_increment.assign((i+1)*dtheta)


    def adaptive(self, F, u, bcu_du, a):

        relaxation = self.parameters["relaxation_parameter"]
        absolute_tol = self.parameters["absolute_tolerance"]
        max_iter = self.parameters["maximum_iterations"]

        vector = u.function_space
        u_temp = Function(u)
        du = Function(u)


        nIter = 0
        eps = 1
        Lnorm = 1e2

        while Lnorm > absolute_tol and nIter < max_iter:
            nIter += 1
            u_temp.assign(self.u)
            A, b = assemble_system(a, -F, bcu_du)
            solve(A, du.vector(), b)
            eps = linalg.norm(du.vector().array(), ord=2)
            omega = 1.0
            self.u.vector()[:] += omega*du.vector()

            L = assemble(self.F)
            for bc in bcu_du:
                bc.apply(L)
                Lnorm2 = norm(L, 'l2')

            while Lnorm2 > Lnorm:
                omega = omega*0.5
                self.u.vector()[:] = u_temp.vector() + omega*du.vector()
                L = assemble(self.F)
                for bc in bcu_du:
                    bc.apply(L)
                    Lnorm2 = norm(L, 'l2')

            Lnorm = Lnorm2
            print '     {0:2d}       {1:3.2E}     {2:1.5E}      {3:1.2E}'.format(nIter,eps, Lnorm2,omega)

