from pyscf.semiempirical import mopac_param as mp

# 1s 2s 2p 3s 3p 3d 4s 4p 
#mp.gexps_1s
elelst = ['H ','He','Li','Be','B ','C ','N ','O ','F ','Ne',
		'Na','Mg','Al','Si','P ','S ','Cl','Ar',
		'K ','Ca','Sc','Ti','V ','Cr','Mn','Fe','Co','Ni','Cu','Zn',
		'Ga','Ge','As','Se','Br','Kr']

with open('sto_6g_basis.dat','w') as wfile:
	wfile.write('BASIS "ao basis" PRINT\n')
	for i, ele in enumerate(elelst):
		if i+1 < 3:
			wfile.write('#BASIS SET: (6s) -> [1s]\n')
			wfile.write(f'{ele}   S\n')
			for i in range(len(mp.gexps_1s)):
				wfile.write(f'{mp.gexps_1s[i]:>20.9e} {mp.gcoefs_1s[i]:>20.9e}\n')
		elif i+1 < 11:
			wfile.write('#BASIS SET: (12s,6p) -> [2s,1p]\n')
			#wfile.write(f'{ele}   S\n')
			#for i in range(len(mp.gexps_1s)):
			#	wfile.write(f'{mp.gexps_1s[i]:>20.9e} {mp.gcoefs_1s[i]:>20.9e}\n')
			wfile.write(f'{ele}   S\n')
			for i in range(len(mp.gexps_2s)):
				wfile.write(f'{mp.gexps_2s[i]:>20.9e} {mp.gcoefs_2s[i]:>20.9e}\n')
			wfile.write(f'{ele}   P\n')
			for i in range(len(mp.gexps_2p)):
				wfile.write(f'{mp.gexps_2p[i]:>20.9e} {mp.gcoefs_2p[i]:>20.9e}\n')
		elif i+1 < 19:
			wfile.write('#BASIS SET: (18s,12p) -> [3s,2p]\n')
			#wfile.write(f'{ele}   S\n')
			#for i in range(len(mp.gexps_1s)):
			#	wfile.write(f'{mp.gexps_1s[i]:>20.9e} {mp.gcoefs_1s[i]:>20.9e}\n')
			#wfile.write(f'{ele}   SP\n')
			#for i in range(len(mp.gexps_2s)):
			#	wfile.write(f'{mp.gexps_2s[i]:>20.9e} {mp.gcoefs_2s[i]:>20.9e}\n')
			#wfile.write(f'{ele}   SP\n')
			#for i in range(len(mp.gexps_2p)):
			#	wfile.write(f'{mp.gexps_2p[i]:>20.9e} {mp.gcoefs_2p[i]:>20.9e}\n')
			wfile.write(f'{ele}   S\n')
			for i in range(len(mp.gexps_3s)):
				wfile.write(f'{mp.gexps_3s[i]:>20.9e} {mp.gcoefs_3s[i]:>20.9e}\n')
			wfile.write(f'{ele}   P\n')
			for i in range(len(mp.gexps_3p)):
				wfile.write(f'{mp.gexps_3p[i]:>20.9e} {mp.gcoefs_3p[i]:>20.9e}\n')
		else:
			wfile.write('#BASIS SET: (24s,18p,6d) -> [4s,3p,1d]\n')
			#wfile.write(f'{ele}   S\n')
			#for i in range(len(mp.gexps_1s)):
			#	wfile.write(f'{mp.gexps_1s[i]:>20.9e} {mp.gcoefs_1s[i]:>20.9e}\n')
			#wfile.write(f'{ele}   SP\n')
			#for i in range(len(mp.gexps_2s)):
			#	wfile.write(f'{mp.gexps_2s[i]:>20.9e} {mp.gcoefs_2s[i]:>20.9e}\n')
			#wfile.write(f'{ele}   SP\n')
			#for i in range(len(mp.gexps_2p)):
			#	wfile.write(f'{mp.gexps_2p[i]:>20.9e} {mp.gcoefs_2p[i]:>20.9e}\n')
			#wfile.write(f'{ele}   SP\n')
			#for i in range(len(mp.gexps_3s)):
			#	wfile.write(f'{mp.gexps_3s[i]:>20.9e} {mp.gcoefs_3s[i]:>20.9e}\n')
			#wfile.write(f'{ele}   SP\n')
			#for i in range(len(mp.gexps_3p)):
			#	wfile.write(f'{mp.gexps_3p[i]:>20.9e} {mp.gcoefs_3p[i]:>20.9e}\n')
			wfile.write(f'{ele}   S\n')
			for i in range(len(mp.gexps_4s)):
				wfile.write(f'{mp.gexps_4s[i]:>20.9e} {mp.gcoefs_4s[i]:>20.9e}\n')
			wfile.write(f'{ele}   P\n')
			for i in range(len(mp.gexps_4p)):
				wfile.write(f'{mp.gexps_4p[i]:>20.9e} {mp.gcoefs_4p[i]:>20.9e}\n')
			#wfile.write(f'{ele}   D\n')
			#for i in range(len(mp.gexps_3d)):
			#	wfile.write(f'{mp.gexps_3d[i]:>20.9e} {mp.gcoefs_3d[i]:>20.9e}\n')

