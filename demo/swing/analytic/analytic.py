__author__ = "Anders Logg"
__copyright__ = "Copyright (C) 2012 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# First added:  2012-03-04
# Last changed: 2012-03-05

from cbc.swing import *
from right_hand_sides import *

# Read parameters
application_parameters = read_parameters()

# For testing
application_parameters["solve_dual"] = False
application_parameters["estimate_error"] = False
application_parameters["plot_solution"] = True
application_parameters["initial_timestep"] = 0.01
application_parameters["uniform_timestep"] = True
application_parameters["output_directory"] = "results"
application_parameters["fixedpoint_tolerance"] = 1e-6

# Define boundaries
noslip  = "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS"
fixed   = "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS || x[1] < DOLFIN_EPS"

# Define structure subdomain
class Structure(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] < 0.5 + DOLFIN_EPS

class Analytic(FSI):

    def __init__(self):

        # Create mesh
        n = 16
        mesh = UnitSquare(n, n)

        # Create analytic expressions
        C = 0.1
        self.f_F = Expression(cpp_f_F)
        self.F_S = Expression(cpp_F_S)
        self.F_M = Expression(cpp_F_M)
        self.p_F = Expression(cpp_p_F)
        self.f_F.C = C
        self.F_S.C = C
        self.F_M.C = C

        # Initialize base class
        FSI.__init__(self, mesh)

    #--- Common ---

    def end_time(self):
        return 0.5

    def evaluate_functional(self, u_F, p_F, U_S, P_S, U_M, dx_F, dx_S, dx_M):
        return U_S[1] * dx_S

    def update(self, t0, t1, dt):
        t = 0.5*(t0 + t1)
        self.f_F.t = t
        self.F_S.t = t
        self.F_M.t = t
        self.p_F.t = t

    def __str__(self):
        return "Channel flow with an immersed elastic flap"

    #--- Fluid problem ---

    def fluid_density(self):
        return 1.0

    def fluid_viscosity(self):
        return 1.0

    def fluid_velocity_dirichlet_values(self):
        return [(0.0, 0.0)]

    def fluid_velocity_dirichlet_boundaries(self):
        return [noslip]

    def fluid_pressure_dirichlet_values(self):
        return [self.p_F]

    def fluid_pressure_dirichlet_boundaries(self):
        return [noslip]

    def fluid_velocity_initial_condition(self):
        return (0.0, 0.0)

    def fluid_pressure_initial_condition(self):
        return 0.0

    def fluid_body_force(self):
        return self.f_F

    #--- Structure problem ---

    def structure(self):
        return Structure()

    def structure_density(self):
        return 10.0

    def structure_mu(self):
        return 1.0

    def structure_lmbda(self):
        return 2.0

    def structure_dirichlet_values(self):
        return [(0.0, 0.0)]

    def structure_dirichlet_boundaries(self):
        return [fixed]

    def structure_neumann_boundaries(self):
        return "on_boundary"

    def structure_body_force(self):
        return self.F_S

    #--- Parameters for mesh problem ---

    def mesh_mu(self):
        return 1.0

    def mesh_lmbda(self):
        return 2.0

    def mesh_alpha(self):
        return 1.0

    def mesh_right_hand_side(self):
        return self.F_M

# Define and solve problem
problem = Analytic()
problem.solve(application_parameters)
