__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *

class FishyFlow(Hyperelasticity):

    def mesh(self):
        mesh = Mesh("dolfin.xml.gz")
        return mesh

    def end_time(self):
        return 10.0

    def time_step(self):
        return 0.1

    def is_dynamic(self):
        return True

    def time_stepping(self):
        return "CG1"

    def neumann_conditions(self):
        flow_push = Expression(("force", "0.0"), force=0.05)
        return [flow_push]

    def neumann_boundaries(self):
        everywhere = "on_boundary"
        return [everywhere]

    def material_model(self):
        mu    = 30.8461/5.0
        lmbda = 50.76/5.0
        material = StVenantKirchhoff([mu, lmbda])
        return material

    def __str__(self):
        return "A hyperelastic fish being pushed by a flow to the right :)"

# Setup and solve problem
problem = FishyFlow()
print problem
problem.solve()
interactive()
