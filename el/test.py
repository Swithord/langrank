import pandas as pd

data = pd.read_csv('el_updated.csv')
s = ['ara','est','aze','fas','msa','orm']
filtered_df = data[~data['Target lang'].isin(s) & ~data['Transfer lang'].isin(s)]
filtered_df.to_csv('el_updated_removed.csv',index=False)


