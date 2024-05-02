import unittest
import copy
import numpy
import scipy.linalg
import pyscf
from pyscf import gto, scf
from pyscf import semiempirical
from pyscf import tdscf
from pyscf import grad

#mol = pyscf.M(atom=[(8,(0,0,0)),(1,(1.4,0,0)),(1,(0,1.4,0))])
#mol.verbose = 4
#mf = semiempirical.RMINDO3(mol).run(conv_tol=1e-6)
mol = pyscf.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
mol.verbose = 4
mf = semiempirical.UMINDO3(mol).run(conv_tol=1e-6)
