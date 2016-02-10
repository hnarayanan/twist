__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from cbc.twist import *
from sys import argv
""" DEMO - Twisting of a hyperelastic cube """

class Twist(StaticHyperelasticity):
    """ Definition of the hyperelastic problem """
    def mesh(self):
        n = 8
        return UnitCubeMesh(n, n, n)

    # Setting up dirichlet conditions and boundaries
    def dirichlet_values(self):
        clamp = Expression(("0.0", "0.0", "0.0"))
        twist = Expression(("0.0",
                            "y0 + (x[1] - y0) * cos(theta) - (x[2] - z0) * sin(theta) - x[1]",
                            "z0 + (x[1] - y0) * sin(theta) + (x[2] - z0) * cos(theta) - x[2]"),
                           y0=0.5, z0=0.5, theta=pi/6)
        return [clamp, twist]

    def dirichlet_boundaries(self):
        left = "x[0] == 0.0"
        right = "x[0] == 1.0"
        return [left, right]


    # List of material models
    def material_model(self):
        # Material parameters can either be numbers or spatially
        # varying fields. For example,
        mu       = 3.8461
        lmbda    = Expression("x[0]*5.8 + (1 - x[0])*5.7")
        C10 = 0.171; C01 = 4.89e-3; C20 = -2.4e-4; C30 = 5.e-4
        delka = 1.0/sqrt(2.0)
        M = Constant((0.0,1.0,0.0))
        k1 = 1e2; k2 = 1e1


        materials = []
        materials.append(MooneyRivlin({'C1':mu/2, 'C2':mu/2, 'bulk':lmbda}))
        materials.append(StVenantKirchhoff({'mu':mu, 'bulk':lmbda}))
        materials.append(neoHookean({'half_nkT':mu, 'bulk':lmbda}))
        materials.append(Isihara({'C10':C10,'C01':C01,'C20':C20,'bulk':lmbda}))
        materials.append(Biderman({'C10':C10,'C01':C01,'C20':C20,'C30':C30,'bulk':lmbda}))
        materials.append(AnisoTest({'mu1':mu,'mu2':2*mu,'M':M,'bulk':lmbda}))
        materials.append(GasserHolzapfelOgden({'mu':mu,'k1':k1,'k2':k2,'M':M,'bulk':lmbda}))
        materials.append(Ogden({'alpha1':1.3,'alpha2':5.0,'alpha3':-2.0,\
                                'mu1':6.3e5,'mu2':0.012e5,'mu3':-0.1e5}))
        
        try:
            index = int(argv[1])
        except:
            index = 2
        print str(materials[index])
        return materials[index]

    def name_method(self, method):
        self.method = method

    def __str__(self):
        return "A hyperelastic cube twisted by 30 degrees solved by " + self.method



# Setup the problem

twist = Twist()
twist.name_method("DISPLACEMENT BASED FORMULATION")


# Solve the problem
print twist
twist.solve()
