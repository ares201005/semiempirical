from pyscf import gto, scf
#from pyscf import semiempirical
#from pyscf.semiempirical import NDDO
from pyscf.semiempirical import OMX
from pyscf.semiempirical.mopac_param import HARTREE2EV

#mol = gto.M(atom=[(8,(0,0,0)),(1,(0.,1.,0.)),(1,(0.0,0.0,1.0))]) #, spin=1)
#mol = gto.M(atom='ch2o2.xyz')
mol = gto.M(atom='hno.xyz')
#mol = gto.M(atom='co.xyz')
mol.verbose = 4 
mf = OMX(mol,model='OM2').run(conv_tol=1e-6,python_integrals=2)

#print("Enuc:", mf.energy_nuc()*HARTREE2EV)
#print("Eelec:", (mf.e_tot-mf.energy_nuc())*HARTREE2EV)
