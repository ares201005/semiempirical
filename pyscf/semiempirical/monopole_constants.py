import pandas as pd
from pyscf.semiempirical import mopac_param
columns=[ 'MOPAC_DD', 'MOPAC_QQ', 'MOPAC_AM', 'MOPAC_AD', 'MOPAC_AQ']
df = pd.DataFrame([ mopac_param.MOPAC_DD, mopac_param.MOPAC_QQ, mopac_param.MOPAC_AM, mopac_param.MOPAC_AD, mopac_param.MOPAC_AQ])
df_t = df.T
df_t.columns=columns
print(df_t)
df_t.to_csv('monopole_constants.csv')
