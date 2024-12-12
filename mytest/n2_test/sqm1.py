import copy
import time
from pyscf import gto, scf
from pyscf import tdscf
from pyscf.semiempirical import sqm
import numpy as np
#mol = gto.M(atom='an2.xyz',spin=0)
#mf = sqm.RMNDO(mol,model='MNDO',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

#mol = gto.M(atom='bn2.xyz',spin=0)
#mf = sqm.RMNDO(mol,model='MNDO',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

mol = gto.M(atom='n2.xyz',spin=0,symmetry=False)
mol = gto.M(atom='rotn2.xyz',spin=0,symmetry=False)
#print('========== STEP 0 ==========')
#mf = sqm.RMNDO(mol,model='AM1',python_integrals=0,atom_sorted=0).run(conv_tol=1e-6)
#print('========== STEP 1 ==========')
#mf = sqm.RMNDO(mol,model='AM1',python_integrals=1,atom_sorted=1).run(conv_tol=1e-6)
#print('========== STEP 2 ==========')
#mf = sqm.RMNDO(mol,model='AM1',python_integrals=2,atom_sorted=1).run(conv_tol=1e-6)
print('========== STEP 3 ==========')
mol.verbose = 4
#mf = sqm.RMNDO(mol,model='AM1',python_integrals=3,atom_sorted=1).run(conv_tol=100) # (conv_tol=1e-6)
mf = sqm.RMNDO(mol,model='AM1',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)


#mf = sqm.RMNDO(mol,model='PM3',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

#mol.verbose = 4
#print('\nH2O RMINDO/3 ground state energy\n')
#mf = sqm.RMINDO3(mol,model='MINDO3',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

#print('\nH2O RMNDO ground state energy\n')
#start = time.perf_counter()
#mf = sqm.RMNDO(mol,model='MNDO',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
#finish = time.perf_counter()
#print('TIME:\t',finish-start)

#print('\nH2O RMNDO-AM1 ground state energy\n')
#mf = sqm.RMNDO(mol,model='AM1',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
#print('\nH2O RMNDO-PM3 ground state energy\n')
#mf = sqm.RMNDO(mol,model='PM3',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
#mol = gto.M(atom=[(8,(0,0,0)),(1,(1.,0,0))], spin=1)
##mol.verbose = 4
#print('\nH2O UMINDO/3 ground state energy\n')
#mf = sqm.UMINDO3(mol,model='MINDO3',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
#print('\nH2O UMNDO ground state energy\n')
#mf = sqm.UMNDO(mol,model='MNDO',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
#print('\nH2O UMNDO-AM1 ground state energy\n')
#mf = sqm.UMNDO(mol,model='AM1',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
#print('\nH2O UMNDO-PM3 ground state energy\n')
#mf = sqm.UMNDO(mol,model='PM3',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)

#print('\nTDHF excited states\n')
#tdA = mf.TDHF().run(nstates=3)

