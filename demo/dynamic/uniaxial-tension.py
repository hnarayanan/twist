__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *

class UniaxialTension(Hyperelasticity):

    def mesh(self):
        n = 8
        return UnitCubeMesh(n, n, n)

    def end_time(self):
        return 1.0

    def time_step(self):
        return 0.1

    def is_dynamic(self):
        return True

    def neumann_conditions(self):
        pull_left   = Expression(("force*t", "0.0", "0.0"), force=-1.0, t=0.0)
        pull_right  = Expression(("force*t", "0.0", "0.0"), force=1.0, t=0.0)
        return [pull_right, pull_left]

    def neumann_boundaries(self):
        left  = "x[0] == 0.0"
        right = "x[0] == 1.0"
        return [right, left]

    def material_model(self):
        mu    = 3.8461
        lmbda = 5.76
        material = LinearElastic([mu, lmbda])
        return material

    def time_stepping(self):
        return "CG1"

    def __str__(self):
        return "A hyperelastic cube being pulled from both sides"

# Setup the problem
problem = UniaxialTension()

# Solve the problem
print problem
problem.solve()
