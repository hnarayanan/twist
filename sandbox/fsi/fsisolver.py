__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Last changed: 2010-06-29

__all__ = ["FSISolver"]

from dolfin import *
from cbc.common import CBCSolver

from primalsolver import solve_primal
from dualsolver import solve_dual
from adaptivity import refine_mesh

class FSISolver(CBCSolver):

    def __init__(self, problem):
        "Initialize FSI solver"

        # Initialize base class
        CBCSolver.__init__(self)

        # Set up parameters
        self.parameters = Parameters("solver_parameters")
        self.parameters.add("plot_solution", True)
        self.parameters.add("save_solution", False)
        self.parameters.add("store_solution_data", False)

    def solve(self, tolerance):
        "Solve the FSI problem"

        # Adaptive loop
        while True:

            # Solve primal problem
            begin("Solving primal problem")
            solve_primal()
            end()

            # Solve dual problem
            begin("Solving dual problem")
            solve_dual()
            end()

            # Estimate error and compute error indicators
            begin("Estimating error and computing error indicators")
            error, indicators = self._estimate_error()
            end()

            # Check if error is small enough
            begin("Checking error estimate")
            if error < tolerance:
                info_green("Adaptive solver converged: error = %g < tolerance = %g" % (error, tolerance))
                break
            end()

            # Refine mesh
            begin("Refining mesh")
            mesh = None
            mesh = refine_mesh(mesh, indicators)
            end()

    def _solve_primal(self):
        "Solve primal problem"

        begin("Solving primal problem")

        # Write solver here

        end()

    def _solve_dual(self):
        "Solve dual problem"

        begin("Solving dual problem")

        # Write solver here

        end()

    def _estimate_error(self):
        "Estimate error and compute error indicators"

        # Compute error indicators
        indicators = []

        # Compute error estimate
        error = 1.0

        return error, indicators

    def _refine_mesh(indicators):
        "Refine mesh"

        pass