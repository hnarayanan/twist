__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *

class Gravity(Hyperelasticity):

    def mesh(self):
        n = 8
        return UnitCubeMesh(n, n, n)

    def end_time(self):
        return 1.0

    def time_step(self):
        return 0.1

    def time_stepping(self):
        return "CG1"

    def dirichlet_values(self):
        clamp = Expression(("0.0", "0.0", "0.0"))
        return [clamp, clamp]

    def dirichlet_boundaries(self):
        left = "x[0] == 0.0"
        right = "x[0] == 1.0"
        return [left, right]

    def body_force(self):
        B = Expression(("0.0", "0.0", "g*t"), g=-9.81, t=0.0)
        return B

    def material_model(self):
        # Material parameters can either be numbers or spatially
        # varying fields. For example,
        mu       = 3.8461
        lmbda    = 5.75
        material = StVenantKirchhoff([mu, lmbda])
        return material

    def __str__(self):
        return "Time-dependent gravity"

# Setup the problem
gravity = Gravity()

# Solve the problem
print gravity
u = gravity.solve()
