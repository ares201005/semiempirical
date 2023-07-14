import re
import numpy as np

with open('ECPSET.f','r') as ofile:
	rfile = ofile.readlines()
switch = 0
elements = []
atommat = []

for line in rfile:
	if re.search('LITHIUM',line) and switch == 0:
		switch += 1
		line = line.split()
		elements.append(line[-1])
		elst = []
		cslst = []
		cplst = []
	elif re.search('ORIGINAL 4G EXPANSIONS',line):
		switch = 1000
		atommat.append([elst,cslst,cplst])
	elif switch == 1:
		if re.search('ES',line):
			line = line.split()
			term = line[-1]
			elst.append(float(term[:-4]))
		elif re.search('CS',line):
			line = line.strip()
			line = line.split()
			term = line[-1]
			cslst.append(float(term[:-4]))
			print(term[:-3])
		elif re.search('CP',line):
			line = line.split()
			term = line[-1]
			cplst.append(float(term[:-4]))
		elif re.search('C \*\*\*',line):
			line = line.split()
			atommat.append([elst,cslst,cplst])
			elst = []
			cslst = []
			cplst = []
			elements.append(line[-1])

for idx, atom in enumerate(elements):
	if atom == 'LITHIUM':
		elements[idx] = 'Li'
	elif atom == 'BERYLLIUM':
		elements[idx] = 'Be'
	elif atom == 'BORON':
		elements[idx] = 'B'
	elif atom == 'CARBON':
		elements[idx] = 'C'
	elif atom == 'NITROGEN':
		elements[idx] = 'N'
	elif atom == 'OXYGEN':
		elements[idx] = 'O'
	elif atom == 'FLUORINE':
		elements[idx] = 'F'
	elif atom == 'NEON':
		elements[idx] = 'Ne'



print(atommat)
with open('tmp_basis.txt','w') as wfile:
	for index, atom in enumerate(elements):
		#wfile.write(f'{atom} nelec 2\n')
		#wfile.write(f'{atom} ul\n')
		wfile.write(f'{atom}\tS\n')
		for jdx in range(0,3):
			wfile.write(f'{atommat[index][0][jdx]:>15.5f}')
			wfile.write(f'{atommat[index][1][jdx]:>15.5f}\n')
		wfile.write(f'{atom}\tP\n')
		for jdx in range(0,3):
			wfile.write(f'{atommat[index][0][jdx]:>15.5f}')
			wfile.write(f'{atommat[index][2][jdx]:>15.5f}\n')
		#for jndex in range(0,3):
			#wfile.write(f'{atommat[index][0][jndex]:>15.5f}')
			#wfile.write(f'{atommat[index][1][jndex]:>15.5f}')
			#wfile.write(f'{atommat[index][2][jndex]:>15.5f}\n')

