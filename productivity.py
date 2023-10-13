import pandas  as pd
from sklearn.preprocessing import LabelEncoder
df = pd.read_json('dataSet_Culture_06102023-POINT.json')
df2 = df.copy()
df2['polygon_x'] = df['polygon'].apply(lambda x: x['x'])
df2['polygon_y'] = df['polygon'].apply(lambda x: x['y'])
df2['month'] = df['Analysis Date'].apply(lambda x: x.split('-')[1])
df2['day'] = df['Analysis Date'].apply(lambda x: x.split('-')[2])
df2['vegetation'] = (df['indextype'] == 'NDVI') & (df['averagevalue'] >= 0.15)
df2 = df2.drop(['polygon', 'soil_id'] , axis = 1)
df2['combined'] = df2['polygon_x'].astype(str) + '_' + df2['polygon_y'].astype(str) + '_' + df2['year contour'].astype(str)
# Assign unique ID based on the grouped column
df2['field_id'] = df2.groupby('combined').ngroup() + 1
# Drop the combined column
df2 = df2.drop(columns=['combined','polygon_x','polygon_y','Analysis Date'])
label_encoder = LabelEncoder()
label_encoder.fit(df2['culture_name'])
df2['culture_name_encoded'] = label_encoder.transform(df2['culture_name'])
encoded_df = pd.get_dummies(df2, columns=['indextype', 'district_name', 'soil_name', 'type_culture_name'])
import h2o
from h2o.automl import H2OAutoML
h2o.init()
encoded_df.to_csv('encoded_data_culture.csv', index=False)
h2odf = h2o.import_file('encoded_data_culture.csv')
h2odf.describe(chunk_summary=True)
train, test = h2odf.split_frame(ratios=[0.8])
aml = H2OAutoML(max_models = 25,
                balance_classes=True)
aml.train(training_frame = train, y = 'culture_name_encoded')
