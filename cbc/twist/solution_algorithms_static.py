__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

# Modified by Anders Logg, 2010
# Last changed: 2012-05-01

from dolfin import *
from nonlinear_solver import *
from cbc.common import *
from cbc.common.utils import *
from cbc.twist.kinematics import Grad, DeformationGradient, Jacobian
from sys import exit
from numpy import array, loadtxt, linalg

def default_parameters():
    "Return default solver parameters."
    p = Parameters("solver_parameters")
    p.add("plot_solution", True)
    p.add("save_solution", False)
    p.add("store_solution_data", False)
    p.add("element_degree",2)
    p.add("problem_formulation",'displacement')
    rel = Parameters("relaxation_parameter")
    rel.add("value", 1.0)
    rel.add("adaptive", True)
    p.add(rel)
    p.add("loading_number_of_steps", 1)

    return p

class StaticMomentumBalanceSolver_U(CBCSolver):
    "Solves the static balance of linear momentum"

    def __init__(self, problem, parameters):
        """Initialise the static momentum balance solver"""

        # Get problem parameters
        mesh = problem.mesh()

        # Define function spaces
        vector = VectorFunctionSpace(mesh, "CG", parameters['element_degree'])
        # Print DOFs
        print "Number of DOFs = %d" % vector.dim()

        # Create boundary conditions
        bcu = create_dirichlet_conditions(problem.dirichlet_values(),
                                          problem.dirichlet_boundaries(),
                                          vector)

        # Define fields
        # Test and trial functions
        v = TestFunction(vector)
        u = Function(vector)
        du = TrialFunction(vector)

        # Driving forces
        B = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        self.theta = Constant(1.0)
        # First Piola-Kirchhoff stress tensor based on the material
        # model
        P  = problem.first_pk_stress(u)
        # The variational form corresponding to hyperelasticity
        L = inner(P, Grad(v))*dx - self.theta*inner(B, v)*dx
        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()

        # If no Neumann conditions are specified, assume it is 0
        if neumann_conditions == []:
            neumann_conditions = [Constant((0,)*vector.mesh().geometry().dim())]

        neumann_boundaries = problem.neumann_boundaries()

        boundary = FacetFunction("size_t", mesh)
        boundary.set_all(len(neumann_boundaries) + 1)



        dsb = Measure('ds')[boundary]
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            compiled_boundary = CompiledSubDomain(neumann_boundary) 
            compiled_boundary.mark(boundary, i)
            L = L - self.theta*inner(neumann_conditions[i], v)*dsb(i)


        a = derivative(L, u, du)

        solver = AugmentedNewtonSolver(L, u, a, bcu,\
                                         load_increment = self.theta)
        solver.parameters["loading_number_of_steps"] \
                    = parameters["loading_number_of_steps"]
        solver.parameters["relaxation_parameter"]["adaptive"] \
                    = parameters["relaxation_parameter"]["adaptive"]
        solver.parameters["relaxation_parameter"]["value"] \
                    = parameters["relaxation_parameter"]["value"]


        self.L = L
        self.du = du
        self.bcu = bcu
        

        # Store parameters
        self.parameters = parameters

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.equation = solver
        self.u = u

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Solve problem
        self.equation.solve()


        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.u, title="Displacement", mode="displacement", rescale=True)
            interactive()

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            displacement_file = File("displacement.xdmf")
            displacement_file << self.u

        # Store solution data
        if self.parameters["store_solution_data"]:
            displacement_series = TimeSeries("displacement")
            displacement_series.store(self.u.vector(), 0.0)

        return self.u

class StaticMomentumBalanceSolver_UP(CBCSolver):
    "Solves the static balance of linear momentum"

    parameters['form_compiler']['representation'] = 'uflacs'
    parameters['form_compiler']['optimize'] = True
    parameters['form_compiler']['quadrature_degree'] = 4

    def __init__(self, problem, parameters):
        """Initialise the static momentum balance solver"""

        # Get problem parameters
        mesh = problem.mesh()

        # Define function spaces
        element_degree = parameters["element_degree"]
        vector = VectorFunctionSpace(mesh, "CG", element_degree)
        scalar = FunctionSpace(mesh,'CG', element_degree - 1)
        mixed_space = MixedFunctionSpace([vector,scalar])

        # Print DOFs
        print "Number of DOFs = %d" % mixed_space.dim()

        # Create boundary conditions
        bcw = create_dirichlet_conditions(problem.dirichlet_values(),
                                          problem.dirichlet_boundaries(),
                                          mixed_space.sub(0))

        # Define fields
        # Test and trial functions
        (v,q) = TestFunctions(mixed_space)
        w = Function(mixed_space)
        (u,p) = split(w)
        dw = TrialFunction(mixed_space)

        # Driving forces
        B = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # First Piola-Kirchhoff stress tensor based on the material
        # model
        P  = problem.first_pk_stress(u)
        J = Jacobian(u)
        F = DeformationGradient(u)
        material_parameters = problem.material_model().parameters
        lb = material_parameters['bulk']

        self.theta = Constant(1.0)
        # The variational form corresponding to hyperelasticity
        L1 = inner(P, Grad(v))*dx - p*J*inner(inv(F.T),Grad(v))*dx - self.theta*inner(B, v)*dx
        L2 = (1.0/lb*p + J - 1.0)*q*dx
        L = L1 + L2

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()

        # If no Neumann conditions are specified, assume it is 0
        if neumann_conditions == []:
            neumann_conditions = Constant((0,)*vector.mesh().geometry().dim())

        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        dsb = ds[boundary]
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            compiled_boundary = CompiledSubDomain(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - self.theta*inner(neumann_conditions[i], v)*dsb(i)


        a = derivative(L, w, dw)

        solver = AugmentedNewtonSolver(L, w, a, bcw,\
                                         load_increment = self.theta)
        solver.parameters["loading_number_of_steps"] \
                    = parameters["loading_number_of_steps"]
        solver.parameters["relaxation_parameter"]["adaptive"] \
                    = parameters["relaxation_parameter"]["adaptive"]
        solver.parameters["relaxation_parameter"]["value"] \
                    = parameters["relaxation_parameter"]["value"]


        # Store parameters
        self.parameters = parameters

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.equation = solver
        (u, p) = w.split()
        self.u = u
        self.p = p

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Solve problem
        self.equation.solve()


        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.u, title="Displacement", mode="displacement", rescale=True)
            plot(self.p, title="Pressure", rescale=True)
            interactive()

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            displacement_file = File("displacement.xdmf")
            pressure_file = File("pressure.xdmf")
            displacement_file << self.u
            pressure_file << self.p

        # Store solution data
        if self.parameters["store_solution_data"]:
            displacement_series = TimeSeries("displacement")
            pressure_series = TimeSeries("pressure")
            displacement_series.store(self.u.vector(), 0.0)
            pressure_series.store(self.p.vector(),0.0)

        return self.u, self.p


class StaticMomentumBalanceSolver_Incompressible(CBCSolver):
    "Solves the static balance of linear momentum"

    parameters['form_compiler']['representation'] = 'uflacs'
    parameters['form_compiler']['optimize'] = True
    parameters['form_compiler']['quadrature_degree'] = 4

    def __init__(self, problem, parameters):
        """Initialise the static momentum balance solver"""

        # Get problem parameters
        mesh = problem.mesh()

        # Define function spaces
        element_degree = parameters["element_degree"]
        vector = VectorFunctionSpace(mesh, "CG", element_degree)
        scalar = FunctionSpace(mesh,'CG', element_degree - 1)
        mixed_space = MixedFunctionSpace([vector,scalar])

        # Print DOFs
        print "Number of DOFs = %d" % mixed_space.dim()

        # Create boundary conditions
        bcw = create_dirichlet_conditions(problem.dirichlet_values(),
                                          problem.dirichlet_boundaries(),
                                          mixed_space.sub(0))

        # Define fields
        # Test and trial functions
        (v,q) = TestFunctions(mixed_space)
        w = Function(mixed_space)
        (u,p) = split(w)
        dw = TrialFunction(mixed_space)

        # Driving forces
        B = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # First Piola-Kirchhoff stress tensor based on the material
        # model
        P  = problem.first_pk_stress(u)
        J = Jacobian(u)
        F = DeformationGradient(u)
        material_parameters = problem.material_model().parameters

        # The variational form corresponding to hyperelasticity
        self.theta = Constant(1.0)
        L1 = inner(P, Grad(v))*dx - p*J*inner(inv(F.T),Grad(v))*dx - self.theta*inner(B, v)*dx
        L2 = (J - 1.0)*q*dx
        L = L1 + L2

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()

        # If no Neumann conditions are specified, assume it is 0
        if neumann_conditions == []:
            neumann_conditions = Constant((0,)*vector.mesh().geometry().dim())

        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        dsb = ds[boundary]
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            compiled_boundary = CompiledSubDomain(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - self.theta*inner(neumann_conditions[i], v)*dsb(i)

        a = derivative(L, w, dw)

        solver = AugmentedNewtonSolver(L, w, a, bcw,\
                                         load_increment = self.theta)
        solver.parameters["loading_number_of_steps"] \
                    = parameters["loading_number_of_steps"]
        solver.parameters["relaxation_parameter"]["adaptive"] \
                    = parameters["relaxation_parameter"]["adaptive"]
        solver.parameters["relaxation_parameter"]["value"] \
                    = parameters["relaxation_parameter"]["value"]


        # Store parameters
        self.parameters = parameters

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.equation = solver
        (u, p) = w.split()
        self.u = u
        self.p = p

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Solve problem
        self.equation.solve()

        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.u, title="Displacement", mode="displacement", rescale=True)
            plot(self.p, title="Pressure", rescale=True)
            interactive()

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            displacement_file = File("displacement.xdmf")
            pressure_file = File("pressure.xdmf")
            displacement_file << self.u
            pressure_file << self.p

        # Store solution data
        if self.parameters["store_solution_data"]:
            displacement_series = TimeSeries("displacement")
            pressure_series = TimeSeries("pressure")
            displacement_series.store(self.u.vector(), 0.0)
            pressure_series.store(self.p.vector(),0.0)

        return self.u, self.p
