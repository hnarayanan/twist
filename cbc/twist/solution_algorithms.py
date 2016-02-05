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
            neumann_conditions = Constant((0,)*vector.mesh().geometry().dim())

        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)


        dsb = ds[boundary]
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            compiled_boundary = CompiledSubDomain(neumann_boundary)
 
            compiled_boundary.mark(boundary, i)
            L = L - self.theta*inner(neumann_conditions[i], v)*dsb(i)
        #plot(boundary,interactive=True)

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
        bcu = create_dirichlet_conditions(problem.dirichlet_values(),
                                          problem.dirichlet_boundaries(),
                                          mixed_space.sub(0))

        # Define fields
        # Test and trial functions
        (v,q) = TestFunctions(mixed_space)
        w = Function(mixed_space)
        (u,p) = split(w)

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

        self.L = L
        self.a = derivative(L, w)
        self.w = w
        self.bcu = bcu

        # Store parameters
        self.parameters = parameters

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = mesh

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Solve problem
        if self.parameters["loading_number_of_steps"] == 1:
            solver = AugmentedNewtonSolver(self.L, self.w, self.bcu)
        else:
            solver = AugmentedNewtonSolver(self.L, self.w, self.bcu,\
                                         load_increment = self.theta)
            solver.parameters["loading_number_of_steps"] \
                    = self.parameters["loading_number_of_steps"]
        solver.parameters["relaxation_parameter"]["adaptive"] \
                    = self.parameters["relaxation_parameter"]["adaptive"]
        solver.parameters["relaxation_parameter"]["value"] \
                    = self.parameters["relaxation_parameter"]["value"]
        solver.solve()

        (u,p) = self.w.split()
        self.u = u
        self.p = p

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
        bcu = create_dirichlet_conditions(problem.dirichlet_values(),
                                          problem.dirichlet_boundaries(),
                                          mixed_space.sub(0))

        # Define fields
        # Test and trial functions
        (v,q) = TestFunctions(mixed_space)
        w = Function(mixed_space)
        (u,p) = split(w)

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

        self.L = L
        self.a = derivative(L, w)
        self.w = w
        self.bcu = bcu

        # Store parameters
        self.parameters = parameters

        # Store variables needed for time-stepping
        # FIXME: Figure out why I am needed
        self.mesh = mesh

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Solve problem
        if self.parameters["loading_number_of_steps"] == 1:
            solver = AugmentedNewtonSolver(self.L, self.w, self.bcu)
        else:
            solver = AugmentedNewtonSolver(self.L, self.w, self.bcu,\
                                         load_increment = self.theta)
            solver.parameters["loading_number_of_steps"] \
                    = self.parameters["loading_number_of_steps"]
        solver.parameters["relaxation_parameter"]["adaptive"] \
                    = self.parameters["relaxation_parameter"]["adaptive"]
        solver.parameters["relaxation_parameter"]["value"] \
                    = self.parameters["relaxation_parameter"]["value"]
        solver.solve()

        (u,p) = self.w.split()
        self.u = u
        self.p = p

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


class MomentumBalanceSolver(CBCSolver):
    "Solves the quasistatic/dynamic balance of linear momentum"

    def __init__(self, problem, parameters):

        """Initialise the momentum balance solver"""

        # Get problem parameters
        mesh        = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        end_time    = problem.end_time()

        # Define function spaces
        element_degree = parameters["element_degree"]
        scalar = FunctionSpace(mesh, "CG", element_degree)
        vector = VectorFunctionSpace(mesh, "CG", element_degree)

        # Get initial conditions
        u0, v0 = problem.initial_conditions()

        # If no initial conditions are specified, assume they are 0
        if u0 == []:
            u0 = Constant((0,)*vector.mesh().geometry().dim())
        if v0 == []:
            v0 = Constant((0,)*vector.mesh().geometry().dim())

        # If either are text strings, assume those are file names and
        # load conditions from those files
        if isinstance(u0, str):
            info("Loading initial displacement from file.")
            file_name = u0
            u0 = Function(vector)
            u0.vector()[:] = loadtxt(file_name)[:]
        if isinstance(v0, str):
            info("Loading initial velocity from file.")
            file_name = v0
            v0 = Function(vector)
            v0.vector()[:] = loadtxt(file_name)[:]

        # Create boundary conditions
        dirichlet_values = problem.dirichlet_values()
        bcu = create_dirichlet_conditions(dirichlet_values,
                                          problem.dirichlet_boundaries(),
                                          vector)

        # Define fields
        # Test and trial functions
        v  = TestFunction(vector)
        u1 = Function(vector)
        v1 = Function(vector)
        a1 = Function(vector)
        du = TrialFunction(vector)

        # Initial displacement and velocity
        u0 = interpolate(u0, vector)
        v0 = interpolate(v0, vector)
        v1 = interpolate(v0, vector)

        # Driving forces
        B  = problem.body_force()

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # Parameters pertinent to (HHT) time integration
        # alpha = 1.0
        beta = 0.25
        gamma = 0.5

        # Determine initial acceleration
        a0 = TrialFunction(vector)
        P0 = problem.first_pk_stress(u0)
        a_accn = inner(a0, v)*dx
        L_accn = - inner(P0, Grad(v))*dx + inner(B, v)*dx

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
            L_accn = L_accn + inner(neumann_conditions[i], v)*dsb(i)

#        problem_accn = LinearVariationalProblem(a_accn, L_accn, a0)
#        solver_accn = LinearVariationalSolver(problem_accn)
#        solver_accn.solve()



#        problem_accn = VariationalProblem(a_accn, L_accn)
#        a0 = problem_accn.solve()

        k = Constant(dt)
        a1 = a0*(1.0 - 1.0/(2*beta)) - (u0 - u1 + k*v0)/(beta*k**2)

        # Get reference density
        rho0 = problem.reference_density()

        # If no reference density is specified, assume it is 1.0
        if rho0 == []:
            rho0 = Constant(1.0)

        density_type = str(rho0.__class__)
        if not ("dolfin" in density_type):
            info("Converting given density to a DOLFIN Constant.")
            rho0 = Constant(rho0)

        # Piola-Kirchhoff stress tensor based on the material model
        P = problem.first_pk_stress(u1)

#         # FIXME: A general version of the trick below is what should
#         # be used instead. The commentend-out lines only work well for
#         # quadratically nonlinear models, e.g. St. Venant Kirchhoff.

#         # S0 = problem.second_pk_stress(u0)
#         # S1 = problem.second_pk_stress(u1)
#         # Sm = 0.5*(S0 + S1)
#         # Fm = DeformationGradient(0.5*(u0 + u1))
#         # P  = Fm*Sm

        # The variational form corresponding to hyperelasticity
        L = int(problem.is_dynamic())*rho0*inner(a1, v)*dx \
            + inner(P, Grad(v))*dx - inner(B, v)*dx

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        dsb = ds[boundary]
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            info("Applying Neumann boundary condition.")
            info(str(neumann_boundary))
            compiled_boundary = CompiledSubDomain(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - inner(neumann_conditions[i], v)*dsb(i)

        a = derivative(L, u1, du)

        # Store variables needed for time-stepping
        self.dt = dt
        self.k = k
        self.t_range = t_range
        self.end_time = end_time
        self.a = a
        self.L = L
        self.bcu = bcu
        self.u0 = u0
        self.v0 = v0
        self.a0 = a0
        self.u1 = u1
        self.v1 = v1
        self.a1 = a1
        self.k  = k
        self.beta = beta
        self.gamma = gamma
        self.vector = vector
        self.B = B
        self.dirichlet_values = dirichlet_values
        self.neumann_conditions = neumann_conditions

        # FIXME: Figure out why I am needed
        self.mesh = mesh
        self.t = 0

        # Empty file handlers / time series
        self.displacement_file = None
        self.velocity_file = None
        self.displacement_series = None
        self.velocity_series = None

        # Store parameters
        self.parameters = parameters

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Time loop
        for t in self.t_range:
            info("Solving the problem at time t = " + str(self.t))
            self.step(self.dt)
            self.update()

        if self.parameters["plot_solution"]:
            interactive()

        return self.u1

    def step(self, dt):
        """Setup and solve the problem at the current time step"""

        # Update time step
        self.dt = dt
        self.k.assign(dt)

        # FIXME: Setup all stuff in the constructor and call assemble instead of VariationalProblem
        problem = NonlinearVariationalProblem(self.L, self.u1, self.bcu, self.a)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-12
        solver.parameters["newton_solver"]["maximum_iterations"] = 100
        solver.solve()
        return self.u1

    def update(self):
        """Update problem at time t"""

        # Compute new accelerations and velocities based on new
        # displacement
        a1 = self.a0*(1.0 - 1.0/(2*self.beta)) \
            - (self.u0 - self.u1 + self.k*self.v0)/(self.beta*self.k**2)
        self.a1 = project(a1, self.vector)
        v1 = self.v0 + self.k*((1 - self.gamma)*self.a1 + self.gamma*self.a0)
        self.v1 = project(v1, self.vector)

        # Propagate the displacements, velocities and accelerations
        self.u0.assign(self.u1)
        self.v0.assign(self.v1)
        self.a0.assign(self.a1)

        # Plot solution
        if self.parameters["plot_solution"]:
            plot(self.u0, title="Displacement", mode="displacement", rescale=True)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.displacement_file is None: self.displacement_file = File("displacement.pvd")
            if self.velocity_file is None: self.velocity_file = File("velocity.pvd")
            self.displacement_file << self.u0
            self.velocity_file << self.v0

        # Store solution data
        if self.parameters["store_solution_data"]:
            if self.displacement_series is None: self.displacement_series = TimeSeries("displacement")
            if self.velocity_series is None: self.velocity_series = TimeSeries("velocity")
            self.displacement_series.store(self.u0.vector(), self.t)
            self.velocity_series.store(self.v0.vector(), self.t)

        # Move to next time step
        self.t = self.t + self.dt

        # Inform time-dependent functions of new time
        for bc in self.dirichlet_values:
            if isinstance(bc, Expression):
                bc.t = self.t
        for bc in self.neumann_conditions:
            bc.t = self.t
        self.B.t = self.t

    def solution(self):
        "Return current solution values"
        return self.u1

class CG1MomentumBalanceSolver(CBCSolver):
    """Solves the dynamic balance of linear momentum using a CG1
    time-stepping scheme"""

    def __init__(self, problem, parameters):

        """Initialise the momentum balance solver"""

        # Get problem parameters
        mesh        = problem.mesh()
        dt, t_range = timestep_range_cfl(problem, mesh)
        end_time    = problem.end_time()
        info("Using time step dt = %g" % dt)

        # Define function spaces
        element_degree = parameters["element_degree"]
        scalar = FunctionSpace(mesh, "CG", element_degree)
        vector = VectorFunctionSpace(mesh, "CG", element_degree)

        mixed_element = MixedFunctionSpace([vector, vector])
        V = TestFunction(mixed_element)
        dU = TrialFunction(mixed_element)
        U = Function(mixed_element)
        U0 = Function(mixed_element)

        # Get initial conditions
        u0, v0 = problem.initial_conditions()

        # If no initial conditions are specified, assume they are 0
        if u0 == []:
            u0 = Constant((0,)*vector.mesh().geometry().dim())
        if v0 == []:
            v0 = Constant((0,)*vector.mesh().geometry().dim())

        # If either are text strings, assume those are file names and
        # load conditions from those files
        if isinstance(u0, str):
            info("Loading initial displacement from file.")
            file_name = u0
            u0 = Function(vector)
            _u0 = loadtxt(file_name)[:]
            u0.vector()[0:len(_u0)] = _u0[:]
        if isinstance(v0, str):
            info("Loading initial velocity from file.")
            file_name = v0
            v0 = Function(vector)
            _v0 = loadtxt(file_name)[:]
            v0.vector()[0:len(_v0)] = _v0[:]

        # Create boundary conditions
        dirichlet_values = problem.dirichlet_values()
        bcu = create_dirichlet_conditions(dirichlet_values,
                                          problem.dirichlet_boundaries(),
                                          mixed_element.sub(0))

        # Functions
        xi, eta = split(V)
        u, v = split(U)
        u_plot = Function(vector)

        # Project u0 and v0 into U0
        a_proj = inner(dU, V)*dx
        L_proj = inner(u0, xi)*dx + inner(v0, eta)*dx
        solve(a_proj == L_proj, U0)
        u0, v0 = split(U0)

        # Driving forces
        B  = problem.body_force()
        if B == []: B = problem.body_force_u(0.5*(u0 + u))

        # If no body forces are specified, assume it is 0
        if B == []:
            B = Constant((0,)*vector.mesh().geometry().dim())

        # Evaluate displacements and velocities at mid points
        u_mid = 0.5*(u0 + u)
        v_mid = 0.5*(v0 + v)

        # Get reference density
        rho0 = problem.reference_density()

        # If no reference density is specified, assume it is 1.0
        if rho0 == []:
            rho0 = Constant(1.0)

        density_type = str(rho0.__class__)
        if not ("dolfin" in density_type):
            info("Converting given density to a DOLFIN Constant.")
            rho0 = Constant(rho0)

        # Piola-Kirchhoff stress tensor based on the material model
        P = problem.first_pk_stress(u_mid)

        # Convert time step to a DOLFIN constant
        k = Constant(dt)

        # The variational form corresponding to hyperelasticity
        L = rho0*inner(v - v0, xi)*dx + k*inner(P, grad(xi))*dx \
            - k*inner(B, xi)*dx + inner(u - u0, eta)*dx \
            - k*inner(v_mid, eta)*dx

        # Add contributions to the form from the Neumann boundary
        # conditions

        # Get Neumann boundary conditions on the stress
        neumann_conditions = problem.neumann_conditions()
        neumann_boundaries = problem.neumann_boundaries()

        boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        boundary.set_all(len(neumann_boundaries) + 1)

        dsb = ds[boundary]
        for (i, neumann_boundary) in enumerate(neumann_boundaries):
            info("Applying Neumann boundary condition.")
            info(str(neumann_boundary))
            compiled_boundary = CompiledSubDomain(neumann_boundary)
            compiled_boundary.mark(boundary, i)
            L = L - k*inner(neumann_conditions[i], xi)*dsb(i)

        a = derivative(L, U, dU)

        # Store variables needed for time-stepping
        self.dt = dt
        self.k = k
        self.t_range = t_range
        self.end_time = end_time
        self.a = a
        self.L = L
        self.bcu = bcu
        self.U0 = U0
        self.U = U
        self.B = B
        self.dirichlet_values = dirichlet_values
        self.neumann_conditions = neumann_conditions

        # FIXME: Figure out why I am needed
        self.mesh = mesh
        # Kristoffer's fix in order to sync the F and S solvers dt...
        self.t = dt

        # Empty file handlers / time series
        self.displacement_file = None
        self.velocity_file = None
        self.displacement_velocity_series = None
        #self.u_plot = u_plot
	self.uplot = plot(u,mode="displacement",title="Displacement")

        # Store parameters
        self.parameters = parameters

    def solve(self):
        """Solve the mechanics problem and return the computed
        displacement field"""

        # Time loop
        for t in self.t_range:
            info("Solving the problem at time t = " + str(self.t))
            self.step(self.dt)
            self.update()

        if self.parameters["plot_solution"]:
            interactive()

        return self.U.split(True)[0]

    def step(self, dt):
        """Setup and solve the problem at the current time step"""

        # Update time step
        self.dt = dt
        self.k.assign(dt)

        problem = NonlinearVariationalProblem(self.L, self.U, self.bcu, self.a)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-12
        solver.parameters["newton_solver"]["relative_tolerance"] = 1e-12
        solver.parameters["newton_solver"]["maximum_iterations"] = 100
        solver.solve()
        return self.U.split(True)

    def update(self):
        """Update problem at time t"""

        u, v = self.U.split()

        # Propagate the displacements and velocities
        self.U0.assign(self.U)

        # Plot solution
        if self.parameters["plot_solution"]:
            # Copy to a fixed function to trick Viper into not opening
            # up multiple windows
            "THIS ASSIGN DOES NOT WORK FOR SOME REASON!" #self.u_plot.assign(u)
            #plot(u, title="Displacement", mode="displacement", rescale=True)
            "This is a new ploting"
	    self.uplot.plot(u)

        # Store solution (for plotting)
        if self.parameters["save_solution"]:
            if self.displacement_file is None: self.displacement_file = File("displacement.pvd")
            if self.velocity_file is None: self.velocity_file = File("velocity.pvd")
            self.displacement_file << u
            self.velocity_file << v

        # Store solution data
        if self.parameters["store_solution_data"]:
            if self.displacement_velocity_series is None: self.displacement_velocity_series = TimeSeries("displacement_velocity")
            self.displacement_velocity_series.store(self.U.vector(), self.t)

        # Move to next time step
        self.t = self.t + self.dt

        # Inform time-dependent functions of new time
        for bc in self.dirichlet_values:
            if isinstance(bc, Expression):
                bc.t = self.t
        for bc in self.neumann_conditions:
            bc.t = self.t
        self.B.t = self.t

    def solution(self):
        "Return current solution values"
        return self.U.split(True) 
