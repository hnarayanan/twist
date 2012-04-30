__author__ = "Kristoffer Selim and Anders Logg"
__copyright__ = "Copyright (C) 2010 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import Parameters, File, info
from utils import date
import os

def default_parameters():
    "Return default values for solver parameters."

    p = Parameters("solver_parameters");

    p.add("solve_primal", True)
    p.add("solve_dual", True)
    p.add("estimate_error", True)
    p.add("plot_solution", False)
    p.add("save_solution", True)
    p.add("save_series", True)
    p.add("uniform_timestep", False)
    p.add("uniform_mesh", False)
    p.add("dorfler_marking", True)
    p.add("global_storage", False)

    p.add("structure_element_degree", 1)
    p.add("mesh_element_degree", 1)
    p.add("max_num_refinements", 100)
    p.add("tolerance", 0.001)
    p.add("fixedpoint_tolerance", 1e-12)
    p.add("initial_timestep", 0.05)
    p.add("num_initial_refinements", 0)
    p.add("maximum_iterations", 1000)
    p.add("num_smoothings", 50)
    p.add("w_h", 0.45)
    p.add("w_k", 0.45)
    p.add("w_c", 0.1)
    p.add("marking_fraction", 0.5)
    p.add("refinement_algorithm", "regular_cut")
    p.add("crossed_mesh", False)
    p.add("use_exact_solution", False)

    p.add("output_directory", "unspecified")
    p.add("description", "unspecified")

    # Hacks
    p.add("fluid_solver", "ipcs")

    return p

def set_output_directory(parameters, problem, case):
    "Set and create output directory"

    # Set output directory
    if parameters["output_directory"] == "unspecified":
        if case is None:
            parameters["output_directory"] = "results-%s-%s" % (problem, date())
        else:
            parameters["output_directory"] = "results-%s-%s" % (problem, str(case))

    # Create output directory
    dir = parameters["output_directory"]
    if not os.path.exists(dir):
        os.makedirs(dir)

def read_parameters():
    """Read parametrs from file specified on command-line or return
    default parameters if no file is specified"""

    # Read parameters from the command-line
    import sys
    p = default_parameters()
    try:
        file = File(sys.argv[1])
        file >> p
        info("Parameters read from %s." % sys.argv[1])
    except:
        info("No parameters specified, using default parameters.")

    # Set output directory
    if p["output_directory"] == "unspecified":
        p["output_directory"] = "results-%s" % date()

    return p

def store_parameters(parameters):
    "Store parameters to file and return filename"

    # Save to file application_parameters.xml
    file = File("application_parameters.xml")
    file << parameters

    # Save to file <output_directory>/application_parameters.xml
    filename = "%s/application_parameters.xml" % parameters["output_directory"]
    file = File(filename)
    file << parameters

    return filename
