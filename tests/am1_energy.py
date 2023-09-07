from pyscf import gto, scf
#from pyscf import semiempirical
#from pyscf.semiempirical import NDDO
from pyscf.semiempirical import RMNDO
from pyscf.semiempirical.mopac_param import HARTREE2EV

mol = gto.M(atom=[(8,(0,0,0)),(1,(0.,1.,0.)),(1,(0.0,0.0,1.0))]) #, spin=1)
mol.verbose = 4 
#mf = NDDO(mol).run(conv_tol=1e-6)
#mf = NDDO(mol).add_keys(method='AM1')
#mf.run(conv_tol=1e-6)
#mf = NDDO(mol,method='AM1').run(conv_tol=1e-6)
mf = RMNDO(mol,model='AM1').run(conv_tol=1e-6)

#print("Enuc:", mf.energy_nuc()*HARTREE2EV)
#print("Eelec:", (mf.e_tot-mf.energy_nuc())*HARTREE2EV)
