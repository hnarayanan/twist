__author__ = "Marek Netusil"


from cbc.twist import *
import mshr
from sys import argv
from dolfin import *
""" DEMO - Inflation and extension of a cylinder. Frequently used experiment in biomechanics
    of arteries - an artery segment is longitudinaly extended by controlling either the stretching
    force of the stretch value and then it is inflated by an internal pressure.
    In this simulation the deformation is controlled by the prescribed longitudinal extension
    and the internal pressure. 
    WARNING - This demo represents some issues one encounters while solving finite elasticity.
    Basic version of a Newton solver will not do and a careful choice of a relaxation parameter
    is needed. Here the choice is taken care of by an adaptive solver. The default Newton solver
    will not converge. """


def create_cylinder():
    """ Geometry of a cylinder """
    R = 1.0     # Inner radius
    H = 0.3     # Thickness
    L = 10.0    # Length

    geometry = mshr.Cylinder(Point(0.0,0.0,0.0),Point(0.0,0.0,L),R+H,R+H,30) - \
    mshr.Cylinder(Point(0.0,0.0,0.0),Point(0.0,0.0,L),R,R,30)
    mesh = mshr.generate_mesh(geometry, 30)

    return mesh

class Cylinder(StaticHyperelasticity):
    """ Definition of the hyperelastic problem """

    def mesh(self):
        # If there is no file with the cylinder mesh create one.
        try:
            mesh = Mesh("cylinder.xml")
        except:
            print "Creating and saving the cylinder mesh as cylinder.xml"
            mesh = create_cylinder()
            File("cylinder.xml") << mesh
        self.n = FacetNormal(mesh)
        return mesh
    
    # Setting up dirichlet conditions and boundaries
    def dirichlet_values(self):
        clamp = Expression(("0.0","0.0","0.0"))
        pull = Constant((0.0,0.0,3.0))
        return [pull, clamp]

    def dirichlet_boundaries(self):
        top = "x[2] == 10.0"
        bottom = "x[2] == 0.0"
        return [top,bottom]


    # Setting up neumann conditions and boundaries
    def neumann_conditions(self):
        try:
            p = Constant(float(argv[1]))
        except:
            p = Constant(100.0)
        pressure = -p*self.n
        return [pressure]

    def neumann_boundaries(self):        
        inside = "pow(x[0],2) + pow(x[1],2) <= 1 + 1e-5 && on_boundary"
        bottom = "x[2] == 0.0"
        return [inside]
    

    # List of material models
    def material_model(self):
        # Material parameters can either be numbers or spatially
        # varying fields. For example,
        mu       = 1e2
        lmbda    = 1e3
        C10 = 0.171; C01 = 4.89e-3; C20 = -2.4e-4; C30 = 5.e-4
        M = Constant((1.0,0.0,0.0))
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
            index = int(argv[2])
        except:
            index = 2
        print str(materials[index])
        return materials[index]

    def name_method(self, method):
        self.method = method

    def __str__(self):
        return "Inflation of a tube solved by " + self.method



# Setup the problem

tube_adp = Cylinder()
tube_adp.name_method("ADAPTIVE NEWTON - DISPLACEMENT BASED FORMULATION")
parameters = tube_adp.parameters['solver_parameters']
parameters['plot_solution'] = True
parameters['save_solution'] = True
parameters['element_degree'] = 2
parameters['relaxation_parameter']['adaptive'] = True


tube_def = Cylinder()
tube_def.name_method("CLASSIC NEWTON - DISPLACEMENT BASED FORMULATION")
parameters = tube_def.parameters['solver_parameters']
parameters['plot_solution'] = True
parameters['save_solution'] = True
parameters['element_degree'] = 2
parameters['relaxation_parameter']['adaptive'] = False



# Solve the problem
print tube_adp
u_adpt = tube_adp.solve()

print tube_def
u_def = tube_def.solve()
