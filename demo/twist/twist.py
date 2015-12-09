__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *
from sys import argv

class Twist(StaticHyperelasticity):

    def mesh(self):
        n = 8
        return UnitCubeMesh(n, n, n)

    def dirichlet_values(self):
        clamp = Expression(("0.0", "0.0", "0.0"))
        """twist = Expression(("0.0",
                            "y0 + (x[1] - y0) * cos(theta) - (x[2] - z0) * sin(theta) - x[1]",
                            "z0 + (x[1] - y0) * sin(theta) + (x[2] - z0) * cos(theta) - x[2]"),
                           y0=0.5, z0=0.5, theta=pi/3)
        return [clamp, twist]"""
        return [clamp]

    def dirichlet_boundaries(self):
        left = "x[0] == 0.0"
        #right = "x[0] == 1.0"
        #return [left, right]
        return [left]

    def neumann_conditions(self):
        traction = Constant((float(argv[1]),0.0,0.0))
        return [traction]

    def neumann_boundaries(self):
        right = "x[0] == 1.0"
        return [right]

    def body_force(self):
        B = Expression(("0.0", "0.0", "0.0"))
        return B

    def material_model(self):
        # Material parameters can either be numbers or spatially
        # varying fields. For example,
        mu       = 1e2
        lmbda    = 1e5
        C10 = 0.171; C01 = 4.89e-3; C20 = -2.4e-4; C30 = 5.e-4

        # It is also easy to switch material models. Uncomment one of
        # the following lines to see a particular material's response.
        materials = []
        materials.append(MooneyRivlin([mu/2, mu/2, lmbda]))
        materials.append(StVenantKirchhoff([mu, lmbda]))
        materials.append(neoHookean([mu, lmbda]))
        materials.append(Isihara([C10, C01, C20, lmbda]))
        materials.append(Biderman([C10, C01, C20, C30, lmbda]))
        materials.append(Ogden([1.3, 5.0, -2.0, 6.3e5, 0.012e5, -0.1e5]))
        
        index = int(argv[2])
        print str(materials[index])
        return materials[index]

    def name_method(self, method):
        self.method = method

    def __str__(self):
        return "A hyperelastic cube twisted by 60 degrees solved by " + self.method



# Setup the problem
twist_dis = Twist()
twist_dis.name_method("DISPLACEMENT BASED FORMULATION")
twist_dis.parameters['solver_parameters']['plot_solution'] = False
twist_dis.parameters['solver_parameters']['element_degree'] = 2


twist_up = Twist()
twist_up.name_method("MIXED (U,P) FORMULATION")
twist_up.parameters['solver_parameters']['problem_formulation'] = 'mixed_up'
twist_up.parameters['solver_parameters']['plot_solution'] = True


# Solve the problem
print twist_dis
u_dis = twist_dis.solve()


print twist_up
u_up, p_up = twist_up.solve()

V = VectorFunctionSpace(twist_dis.mesh(),'CG',2)

u_e = project(u_up - u_dis,V)
e = assemble(inner(u_e,u_e)*dx)
print "L2 norm of the difference between displacement and (u,p) formulation \n ||u_dis - u_up||^2 = %s"  % e
#plot(u_e, mode = "displacement", interactive=True)

