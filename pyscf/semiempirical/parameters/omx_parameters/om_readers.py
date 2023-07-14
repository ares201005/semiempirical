import numpy as np
import pandas as pd

# OM1
def om1_reader(method_param):
	omx = method_param[0]
	numelem = int(len(omx)/28)
	numparm = 28
	tmp_lst = []
	full_lst = []
	el_lst = ['H','C','N','O','F']
	full_lst.append(el_lst)
	labels = ['element',
		  'U_ss',
		  'U_pp', 
		  'zeta_s',  
		  'zeta_p',  
		  'beta_s',
		  'beta_p',
		  'beta_pi',
		  'beta_sh',
		  'beta_ph',
		  'alpha_s',
		  'alpha_p',
		  'alpha_pi',
		  'alpha_sh',
		  'alpha_ph',
		  'fval1',
		  'fval2',
		  'eisol',
		  'hyf',
		  'g_ss',
		  'g_sp',
		  'g_pp',
		  'g_p2',
		  'h_sp',
		  'dd',
		  'qq',
		  'am',
		  'ad',
		  'aq']
	#labels = ['Element','USS','UPP','ZS','ZP','BETAS',
	#	  'BETAP','HYF','BETPI','BETSH',
	#	  'BETPH','ALPS','ALPP','ALPPI',
	#	  'ALPSH','ALPPH','FVAL','FVAL2',
	#	  'DD','QQ','AM','AD','AQ',
	#	  'GSS','GSP','GPP','GP2',
	#	  'HSP','EISOL']
	for i in range(0,numparm): # for each element 0-5
		for j in omx[i::numparm]: # for each parameter
			tmp_lst.append(j[-1][:-3]) # e.g. USS
		full_lst.append(tmp_lst)
		tmp_lst = []
	df = pd.DataFrame(full_lst)
	zlist = np.zeros((numparm+1))
	df.insert(0,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.columns = [0,1,2,3,4,5,6,7,8,9]
	df2 = df.T
	df2.iloc[2][0] = 'He'
	df2.iloc[3][0] = 'Li'
	df2.iloc[4][0] = 'Be'
	df2.iloc[5][0] = 'B'
	df2.columns = labels
	df2.to_csv('parameters_OM1.csv')


# 34 186
def om2_reader(method_param):
	omx = method_param[1]
	numparm = 38
	numelem = int(len(omx)/numparm)
	tmp_lst = []
	full_lst = []
	el_lst = ['H','C','N','O','F']
	full_lst.append(el_lst)
	labels = ['element',
		  'U_ss',
		  'U_pp', 
		  'zeta_s',  
		  'zeta_p',  
		  'beta_s',
		  'beta_p',
		  'beta_pi',
		  'beta_sh',
		  'beta_ph',
		  'alpha_s',
		  'alpha_p',
		  'alpha_pi',
		  'alpha_sh',
		  'alpha_ph',
		  'fval1',
		  'fval2',
		  'gval1',
		  'gval2',
		  'z_sc',
		  'f_sc',
		  'b_sc',
		  'a_sc',
		  'eisol',
		  'hyf',
		  'g_ss',
		  'g_sp',
		  'g_pp',
		  'g_p2',
		  'h_sp',
		  'dd',
		  'qq',
		  'am',
		  'ad',
		  'aq',
		  'c61',  
		  'r01',  
		  'c62',  
		  'r02']   
	print(len(labels))
	#labels = ['USS','UPP','ZS','ZP','BETAS',
	#	  'BETAP','HYF','BETPI','BETSH',
	#	  'BETPH','ALPS','ALPP','ALPPI',
	#	  'ALPSH','ALPPH','FVAL1','FVAL2',
	#	  'GVAL1','GVAL2',
	#	  'ZSC','FSC','BSC','ASC',
	#	  'DD','QQ','AM','AD','AQ',
	#	  'GSS','GSP','GPP','GP2',
	#	  'HSP','EISOL',
	#	  'C61','R01','C62','R02']
	for i in range(0,numparm): # for each element 0-5
		for j in omx[i::numparm]: # for each parameter
			print(j[-1][:-3])
			tmp_lst.append(j[-1][:-3]) # e.g. USS
		full_lst.append(tmp_lst)
		tmp_lst = []
	df = pd.DataFrame(full_lst)
	zlist = np.zeros((numparm+1))
	df.insert(0,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.columns = [0,1,2,3,4,5,6,7,8,9]
	df2 = df.T
	df2.iloc[2][0] = 'He'
	df2.iloc[3][0] = 'Li'
	df2.iloc[4][0] = 'Be'
	df2.iloc[5][0] = 'B'
	df2.columns = labels
	df2.to_csv('parameters_OM2.csv')

# 34 186
def om3_reader(method_param):
	omx = method_param[2]
	numparm = 38
	numelem = int(len(omx)/numparm)
	print(numelem)
	tmp_lst = []
	full_lst = []
	el_lst = ['H','C','N','O','F']
	full_lst.append(el_lst)
	labels = ['element',
		  'U_ss',
		  'U_pp', 
		  'zeta_s',  
		  'zeta_p',  
		  'beta_s',
		  'beta_p',
		  'beta_pi',
		  'beta_sh',
		  'beta_ph',
		  'alpha_s',
		  'alpha_p',
		  'alpha_pi',
		  'alpha_sh',
		  'alpha_ph',
		  'fval1',
		  'fval2',
		  'gval1',
		  'gval2',
		  'z_sc',
		  'f_sc',
		  'b_sc',
		  'a_sc',
		  'eisol',
		  'hyf',
		  'g_ss',
		  'g_sp',
		  'g_pp',
		  'g_p2',
		  'h_sp',
		  'dd',
		  'qq',
		  'am',
		  'ad',
		  'aq',
		  'c61',  
		  'r01',  
		  'c62',  
		  'r02']   
	#labels = ['USS','UPP','ZS','ZP','BETAS',
	#	  'BETAP','HYF','BETPI','BETSH',
	#	  'BETPH','ALPS','ALPP','ALPPI',
	#	  'ALPSH','ALPPH','FVAL1','FVAL2',
	#	  'GVAL1','GVAL2',
	#	  'ZSC','FSC','BSC','ASC',
	#	  'DD','QQ','AM','AD','AQ',
	#	  'GSS','GSP','GPP','GP2',
	#	  'HSP','EISOL',
	#	  'C61','R01','C62','R02']
	for i in range(0,numparm): # for each element 0-5
		for j in omx[i::numparm]: # for each parameter
			tmp_lst.append(j[-1][:-3]) # e.g. USS
		full_lst.append(tmp_lst)
		tmp_lst = []
	df = pd.DataFrame(full_lst)
	zlist = np.zeros((numparm+1))
	df.insert(0,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.columns = [0,1,2,3,4,5,6,7,8,9]
	df2 = df.T
	df2.iloc[2][0] = 'He'
	df2.iloc[3][0] = 'Li'
	df2.iloc[4][0] = 'Be'
	df2.iloc[5][0] = 'B'
	df2.columns = labels
	df2.to_csv('parameters_OM3.csv')

# 0 

# 29 166
def odm2_reader(method_param):
	omx = method_param[4]
	numparm = 34 # 35?
	numelem = int(len(omx)/numparm)
	print(numelem)
	tmp_lst = []
	full_lst = []
	el_lst = ['H','C','N','O','F']
	full_lst.append(el_lst)
	labels = ['element',
		  'U_ss',
		  'U_pp', 
		  'zeta_s',  
		  'zeta_p',  
		  'beta_s',
		  'beta_p',
		  'beta_pi',
		  'beta_sh',
		  'beta_ph',
		  'alpha_s',
		  'alpha_p',
		  'alpha_pi',
		  'alpha_sh',
		  'alpha_ph',
		  'fval1',
		  'fval2',
		  'gval1',
		  'gval2',
		  'z_sc',
		  'f_sc',
		  'b_sc',
		  'a_sc',
		  'eisol',
		  'hyf',
		  'g_ss',
		  'g_sp',
		  'g_pp',
		  'g_p2',
		  'h_sp',
		  'dd',
		  'qq',
		  'am',
		  'ad',
		  'aq']   
	#labels = ['USS','UPP','ZS','ZP',
	#	  'BETAS','BETAP','HYF','BETPI',
	#	  'BETSH','BETPHD','ALPS','ALPP',
	#	  'ALPI','ALSH','ALPH','FVAL1',
	#	  'FVAL2','GVAL1','GVAL2',
	#	  'ZSC','FSC','BSC','ASC',
	#	  'DD','QQ','AM','AD',
	#	  'AQ','GSS','GSP','GPP',
	#	  'GP2','HSP','EISOL']
	for i in range(0,numparm): # for each element 0-5
		for j in omx[i::numparm]: # for each parameter
			tmp_lst.append(j[-1][:-3]) # e.g. USS
		full_lst.append(tmp_lst)
		tmp_lst = []
	df = pd.DataFrame(full_lst)
	zlist = np.zeros((numparm+1))
	df.insert(0,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.columns = [0,1,2,3,4,5,6,7,8,9]
	df2 = df.T
	df2.iloc[2][0] = 'He'
	df2.iloc[3][0] = 'Li'
	df2.iloc[4][0] = 'Be'
	df2.iloc[5][0] = 'B'
	df2.columns = labels
	df2.to_csv('parameters_ODM2.csv')

# 29 166 idk
def odm3_reader(method_param):
	omx = method_param[5]
	numparm = 34 # 34
	numelem = int(len(omx)/numparm)
	print(numelem)
	tmp_lst = []
	full_lst = []
	el_lst = ['H','C','N','O','F']
	full_lst.append(el_lst)
	labels = ['element',
		  'U_ss',
		  'U_pp', 
		  'zeta_s',  
		  'zeta_p',  
		  'beta_s',
		  'beta_p',
		  'beta_pi',
		  'beta_sh',
		  'beta_ph',
		  'alpha_s',
		  'alpha_p',
		  'alpha_pi',
		  'alpha_sh',
		  'alpha_ph',
		  'fval1',
		  'fval2',
		  'gval1',
		  'gval2',
		  'z_sc',
		  'f_sc',
		  'b_sc',
		  'a_sc',
		  'eisol',
		  'hyf',
		  'g_ss',
		  'g_sp',
		  'g_pp',
		  'g_p2',
		  'h_sp',
		  'dd',
		  'qq',
		  'am',
		  'ad',
		  'aq']   
	#labels = ['USS','UPP','ZS','ZP',
	#	  'BETAS','BETAP','HYF','BETPI',
	#	  'BETSH','BETPH','ALPS','ALPP',
	#	  'ALPI','ALSH','ALPH','FVAL1',
	#	  'FVAL2','GVAL1','GVAL2',
	#	  'ZSC','FSC','BSC','ASC',
	#	  'DD','QQ','AM','AD',
	#	  'AQ','GSS','GSP','GPP',
	#	  'GP2','HSP','EISOL']
	for i in range(0,numparm): # for each element 0-5
		for j in omx[i::numparm]: # for each parameter
			tmp_lst.append(j[-1][:-3]) # e.g. USS
		full_lst.append(tmp_lst)
		tmp_lst = []
	df = pd.DataFrame(full_lst)
	zlist = np.zeros((numparm+1))
	df.insert(0,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.insert(2,column=0,value=zlist,allow_duplicates=True)
	df.columns = [0,1,2,3,4,5,6,7,8,9]
	df2 = df.T
	df2.iloc[2][0] = 'He'
	df2.iloc[3][0] = 'Li'
	df2.iloc[4][0] = 'Be'
	df2.iloc[5][0] = 'B'
	print('Labels',len(labels))
	print('zlist',len(zlist))
	print('lendf2',len(df2.iloc[1]))
	df2.columns = labels
	df2.to_csv('parameters_ODM3.csv')


