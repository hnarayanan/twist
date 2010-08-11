"This module defines special operators for the dual problem and residuals."

__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-08-11

from dolfin import *

# Define identity matrix (2D)
I = Identity(2)

def F(v):
    "Return deformation gradient"
    return I + grad(v)

def J(v):
    "Return determinant of deformation gradient"
    return det(F(v))

def DJ(v, dv):
    "Return derivative J'(v) dv of determinant"
    return dv[0].dx(0)*(1 - v[1].dx(1)) - dv[0].dx(1)*v[1].dx(0) \
          -dv[1].dx(0)*v[0].dx(1) + dv[1].dx(1)*(1 + v[0].dx(0))

def Sigma_F(U_F, P_F, U_M, mu_F):
    "Return fluid stress in reference domain"
    return mu_F*(grad(U_F)*inv(F(U_M)) + inv(F(U_M)).T*grad(U_F).T) - P_F*I

def Sigma_M(U_M, mu_M, lmbda_M):
    "Return mesh stress in reference domain"
    return 2*mu_M*sym(grad(U_M)) + lmbda_M*tr(sym(grad(U_M)))*I
