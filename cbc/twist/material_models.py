__author__ = "Harish Narayanan"
__copyright__ = "Copyright (C) 2009 Simula Research Laboratory and %s" % __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from material_model_base import MaterialModel, MaterialModel_Anisotropic

class LinearElastic(MaterialModel):
    """Defines the strain energy function for a linear elastic
    material"""

    def model_info(self):
        self.num_parameters = 2
        self.kinematic_measure = "InfinitesimalStrain"

    def strain_energy(self, parameters):
        epsilon = self.epsilon
        mu, lmbda = parameters['mu'], parameters['bulk']
        return lmbda/2*(tr(epsilon)**2) + mu*tr(epsilon*epsilon)

class StVenantKirchhoff(MaterialModel):
    """Defines the strain energy function for a St. Venant-Kirchhoff
    material"""

    def model_info(self):
        self.num_parameters = 2
        self.kinematic_measure = "GreenLagrangeStrain"

    def strain_energy(self, parameters):
        E = self.E
        mu, lmbda = parameters['mu'], parameters['bulk']
        return lmbda/2*(tr(E)**2) + mu*tr(E*E)

class MooneyRivlin(MaterialModel):
    """Defines the strain energy function for a (two term)
    Mooney-Rivlin material"""

    def model_info(self):
        self.num_parameters = 3
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        J = sqrt(self.I3)
        I1bar = J**(-2.0/3.0)*self.I1
        I2bar = J**(-4.0/3.0)*self.I2

        C1, C2, bulk = parameters['C1'], parameters['C2'], parameters['bulk']
        return C1*(I1bar - 3) + C2*(I2bar - 3) + bulk*(J - 1.0)**2

class neoHookean(MaterialModel):
    """Defines the strain energy function for a neo-Hookean
    material"""

    def model_info(self):
        self.num_parameters = 2
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        J = sqrt(self.I3)
        I1bar = J**(-2.0/3.0)*self.I1
        I1 = self.I1

        half_nkT, bulk = parameters['half_nkT'], parameters['bulk']
        return half_nkT*(I1bar - 3.0) + bulk*(J - 1.0)**2
        #return half_nkT*(I1 - 3.0)    


class Isihara(MaterialModel):
    """Defines the strain energy function for an Isihara material"""

    def model_info(self):
        self.num_parameters = 4
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        J = sqrt(self.I3)
        I1bar = J**(-2.0/3.0)*self.I1
        I2bar = J**(-4.0/3.0)*self.I2

        C10, C01, C20, bulk = parameters['C10'], parameters['C01'], \
                 parameters['C20'], parameters['bulk']
        return C10*(I1bar - 3) + C01*(I2bar - 3) \
                 + C20*(I2bar - 3)**2 + bulk*(J - 1.0)**2

class Biderman(MaterialModel):
    """Defines the strain energy function for a Biderman material"""

    def model_info(self):
        self.num_parameters = 5
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        J = sqrt(self.I3)
        I1bar = J**(-2.0/3.0)*self.I1
        I2bar = J**(-4.0/3.0)*self.I2

        C10, C01, C20, C30, bulk = parameters['C10'], parameters['C01'], \
                parameters['C20'], parameters['C30'], parameters['bulk']
        return C10*(I1bar - 3) + C01*(I2bar - 3) \
                + C20*(I2bar - 3)**2 + C30(I1bar - 3)**3 + bulk*(J - 1.0)**2

class GentThomas(MaterialModel):
    """Defines the strain energy function for a Gent-Thomas
    material"""

    def model_info(self):
        self.num_parameters = 2
        self.kinematic_measure = "CauchyGreenInvariants"

    def strain_energy(self, parameters):
        I1 = self.I1
        I2 = self.I2

        C1, C2 = parameters['C1'], parameters['C2']
        return C1*(I1 - 3) + C2*ln(I2/3)

class Ogden(MaterialModel):
    """Defines the strain energy function for a (six parameter) Ogden
    material"""

    def model_info(self):
        self.num_parameters = 6
        self.kinematic_measure = "PrincipalStretches"

    def strain_energy(self, parameters):
        l1 = self.l1
        l2 = self.l2
        l3 = self.l3

        alpha1, alpha2, alpha3, mu1, mu2, mu3 = parameters['alpha1'], parameters['alpha2'], \
            parameters['alpha3'], parameters['mu1'], parameters['mu2'], parameters['mu3']
        return mu1/alpha1*(l1**alpha1 + l2**alpha1 + l3**alpha1 - 3) \
            +  mu2/alpha2*(l1**alpha2 + l2**alpha2 + l3**alpha2 - 3) \
            +  mu3/alpha3*(l1**alpha3 + l2**alpha3 + l3**alpha3 - 3)


class AnisoTest(MaterialModel_Anisotropic):
    """Testing model for anisotropic materials.
    psi = mu1(I1 - 3) + mu2(I4 - 1) + bulk(J - 1)^2
    with mu2 representing the stiffness of fibres having the direction M"""
    
    def model_info(self):
        self.num_parameters = 4
        self.kinematic_measure = "AnisotropicInvariants"

    def strain_energy(self, parameters):     
        J = sqrt(self.I3)
        I1bar = J**(-2.0/3.0)*self.I1
        I4 = self.I4

        mu1, mu2, bulk = parameters['mu1'], parameters['mu2'], parameters['bulk']
        return mu1*(I1bar - 3.0) + mu2*(I4 - 1.0)**2 + bulk*(J - 1.0)**2


class GasserHolzapfelOgden(MaterialModel_Anisotropic):
    """ Defines the strain energy function for an anisotropic
    Gasser-Holzapfel-Ogden material """
    def model_info(self):
        self.num_parameters = 5
        self.kinematic_measure = "AnisotropicInvariants"

    def strain_energy(self, parameters):
        J = sqrt(self.I3)
        I1bar = J**(-2.0/3.0)*self.I1
        I4 = self.I4
    
        mu, k1, k2, bulk = parameters['mu'], parameters['k1'], parameters['k2'], parameters['bulk']
        return mu*(I1bar - 3.0) + k1/k2*(exp(k2*(I4 - 1.0)**2) - 1.0) + bulk*(J - 1.0)**2
