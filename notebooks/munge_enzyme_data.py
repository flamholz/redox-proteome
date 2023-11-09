import numpy as np
import pandas as pd

# Load the Bar-Even 2011 data on enzyme kinetics
enz_data = pd.read_excel('../data/enzymes/BarEven2011.xls', sheet_name='1. KineticTable')
enz_data.columns = 'EC1,EC2,EC3,EC4,compound_ID,reaction_ID,direction,organism_ID,publication_ID,T,pH,KM_uM,kcat_s'.split(',')
enz_data.head()

modules = pd.read_excel('../data/enzymes/BarEven2011.xls', sheet_name='6. Metabolic Modules')
modules.columns = 'reaction_ID,module_ID'.split(',')
modules = modules.set_index('reaction_ID')

module_groups = pd.read_excel('../data/enzymes/BarEven2011.xls', sheet_name='module_groups')
module_groups.columns = 'module_ID,module_name,module_type'.split(',')
module_groups = module_groups.set_index('module_ID')

# Join in module information. Default join is 'left' i.e. keep all rows in enz_data.
enz_data = enz_data.join(modules, on='reaction_ID').join(module_groups, on='module_ID')
enz_data.to_csv('../data/enzymes/BarEven2011_compact_kinetics.csv', index=False)