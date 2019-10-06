import json

fileName = 'cat_ord_dict.txt'
f = open(fileName,'r')
my_dict = json.loads(f.read())
print(my_dict)
print(my_dict['LotShape'])



##### How to write a dict in file #####
# cat_ord_dict = {
#     'LotShape': {
#         'IR3': 0,
#         'IR2': 1,
#         'IR1': 2,
#         'Reg': 3
#     }, 
#     'Utilities': {
#         'ELO': 0,
#         'NoSeWa': 1,
#         'NoSewr': 2,
#         'AllPub': 3
#     }, 
#     'LandSlope': {
#         'Gtl': 0,
#         'Mod': 1,
#         'Sev': 2
#     }, 
#     'ExterQual': {
#         'Po': 0,
#         'Fa': 1,
#         'TA': 2,
#         'Gd': 3,
#         'Ex': 4
#     },
#     'ExterCond': {
#         'Ex': 4,
#         'Gd': 3,
#         'TA': 2,
#         'Fa': 1,
#         'Po': 0
#     },
#     'BsmtQual': {
#         'Ex': 5,
#         'Gd': 4,
#         'TA': 3,
#         'Fa': 2,
#         'Po': 1,
#         'NA': 0
#     },
#     'BsmtCond': {
#         'Ex': 5,
#         'Gd': 4,
#         'TA': 3,
#         'Fa': 2,
#         'Po': 1,
#         'NA': 0    
#     },
#     'BsmtExposure': {
#         'Gd': 4,
#         'Av': 3,
#         'Mn': 2,
#         'No': 1,
#         'NA': 0
#     },
#     'BsmtFinType1': {
#         'GLQ': 6,
#         'ALQ': 5,
#         'BLQ': 4,
#         'Rec': 3,
#         'LwQ': 2,
#         'Unf': 1,
#         'NA': 0     
#     },
#     'BsmtFinType2': {
#         'GLQ': 6,
#         'ALQ': 5,
#         'BLQ': 4,
#         'Rec': 3,
#         'LwQ': 2,
#         'Unf': 1,
#         'NA': 0     
#     },
#     'HeatingQC': {
#         'Ex': 4,
#         'Gd': 3,
#         'TA': 2,
#         'Fa': 1,
#         'Po': 0
#     },
#     'KitchenQual': {
#         'Ex': 4,
#         'Gd': 3,
#         'TA': 2,
#         'Fa': 1,
#         'Po': 0
#     },
#     'Functional': {
#         'Typ': 7,
#         'Min1': 6,
#         'Min2': 5,
#         'Mod': 4,
#         'Maj1': 3,
#         'Maj2': 2,
#         'Sev': 1,
#         'Sal': 0
#     },
#     'FireplaceQu': {
#         'Ex': 5,
#         'Gd': 4,
#         'TA': 3,
#         'Fa': 2,
#         'Po': 1,
#         'NA': 0
#     },
#     'GarageFinish': {
#         'Fin': 3,
#         'RFn': 2,
#         'Unf': 1,
#         'NA': 0 
#     },
#     'GarageQual': {
#         'Ex': 5,
#         'Gd': 4,
#         'TA': 3,
#         'Fa': 2,
#         'Po': 1,
#         'NA': 0
#     },
#     'GarageCond': {
#         'Ex': 5,
#         'Gd': 4,
#         'TA': 3,
#         'Fa': 2,
#         'Po': 1,
#         'NA': 0
#     },
#     'PavedDrive': {
#         'Y': 2,
#         'P': 1,
#         'N': 0
#     },
#     'PoolQC': {
#         'Ex': 4,
#         'Gd': 3,
#         'TA': 2,
#         'Fa': 1,
#         'NA': 0
#     },
#     'Fence': {
#         'GdPrv': 4,
#         'MnPrv': 3,
#         'GdWo': 2,
#         'MnWw': 1,   
#         'NA': 0
#     }
# }

# txt = json.dumps(cat_ord_dict)
# f = open('cat_ord_dict.txt','w')
# f.write(txt)

