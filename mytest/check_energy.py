import re, os, glob
from pyscf import gto, scf
from pyscf import tdscf
from pyscf.semiempirical import sqm

mndo_files = glob.glob('mndo2020_comp/*out')
mndo_files.sort()

# am1_water.out

with open('tabulated_comparison.txt','w') as wfile:
	wfile.write(f'{"Model":>10} {"MNDO2020":>16} {"PYSCF SQM":>16}\n')
	for f in mndo_files:
		name = os.path.basename(f)
		model = name[:-10]
		with open(f,'r') as ofile:
			rfile = ofile.readlines()
		for line in rfile:
			if re.search('SCF TOTAL ENERGY',line):
				scf = float(line.split()[-2]) #eV
			elif re.search('ELECTRONIC ENERGY',line):
				ee  = float(line.split()[-2]) #eV
			elif re.search('NUCLEAR ENERGY',line):
				ne  = float(line.split()[-2]) #eV
		mol = gto.M(atom='water1.xyz',spin=0)
		if model == 'mindo3':
			mf = sqm.RMINDO3(mol,model='MINDO3',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
		elif model == 'mndo':
			mf = sqm.RMNDO(mol,model='MNDO',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
		elif model == 'am1':
			mf = sqm.RMNDO(mol,model='AM1',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
		elif model == 'pm3':
			mf = sqm.RMNDO(mol,model='PM3',python_integrals=3,atom_sorted=1).run(conv_tol=1e-6)
			#mf = sqm.RMNDO(mol,model='PM3',python_integrals=1).run(conv_tol=1e-6)
		print(model)
		py_scf = mf.e_tot * 27.21
		py_ne  = mf.energy_nuc() * 27.21
		py_ee0 = mf.energy_elec()[0] * 27.21
		py_ee = py_scf - py_ne
		wfile.write(f'{model.upper()+" SCF":>10} {scf:>16.8f} {py_scf:>16.5f} {py_scf:>16.8f}\n')
		wfile.write(f'{model.upper()+" EE ":>10} {ee:>16.8f} {py_ee0:>16.5f} {py_ee0:>16.8f}\n')
		wfile.write(f'{model.upper()+" EE ":>10} {ee:>16.8f} {py_ee:>16.5f} {py_ee:>16.8f}\n')
		wfile.write(f'{model.upper()+" NE ":>10} {ne:>16.8f} {py_ne:>16.5f} {py_ne:>16.8f} \n\n')
