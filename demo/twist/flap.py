__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *

class Obstruction(Hyperelasticity):

    def mesh(self):
        n = 4
        return RectangleMesh(0, 0, 0.2, 0.5, n, 5*n/2)

    def end_time(self):
        return 1.0

    def time_step(self):
        return 0.2

    def is_dynamic(self):
        return True

    def neumann_conditions(self):
        fluid_force = Expression(("magnitude*t", "0.0"), magnitude=1.5, t=0)
        return [fluid_force]

    def neumann_boundaries(self):
        fluid_interface = "x[1] > 0.0 && x[0] == 0"
        return [fluid_interface]

    def dirichlet_values(self):
        fix = Constant((0.0, 0.0))
        return [fix]

    def dirichlet_boundaries(self):
        bottom = "x[1] == 0.0"
        return [bottom]

    def material_model(self):
        mu    = 60
        lmbda = 90
        material = StVenantKirchhoff([mu, lmbda])
        return material

    def reference_density(self):
        return 1.0

    def time_stepping(self):
        return "CG1"

    def __str__(self):
        return "An obstruction being deformed by an ambient flow"

# Setup problem
problem = Obstruction()

# Solve problem
print problem
problem.solve()
