import copy
import time
from pyscf import gto, scf
from pyscf import tdscf
from pyscf.semiempirical import sqm
#mol = gto.M(atom='an2.xyz',spin=0)
#mf = sqm.RMNDO(mol,model='MNDO',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

#mol = gto.M(atom='bn2.xyz',spin=0)
#mf = sqm.RMNDO(mol,model='MNDO',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

mol = gto.M(atom='ch2o.xyz',spin=0,symmetry=False)
print('\nH2O RMINDO/3 ground state energy\n')
mf = sqm.RMINDO3(mol,model='MINDO3',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

