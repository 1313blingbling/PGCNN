# -*- coding: utf-8 -*-
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from mordred import Calculator, PBF, MomentOfInertia, TopologicalCharge, MolecularDistanceEdge, MoRSE, GravitationalIndex, GeometricalIndex, EState, DistanceMatrix, DetourMatrix, CPSA, BaryszMatrix, Autocorrelation, AdjacencyMatrix, descriptors, get_descriptors_from_module
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

descs = get_descriptors_from_module(descriptors, submodule=True)

# exclude some from descs
descs = filter(lambda d: ((d.__module__ != AdjacencyMatrix.__name__) and 
                          (d.__module__ != Autocorrelation.__name__) and
                          (d.__module__ != DetourMatrix.__name__) and 
                          (d.__module__ != BaryszMatrix.__name__) and 
                          (d.__module__ != CPSA.__name__) and 
                          (d.__module__ != DistanceMatrix.__name__) and 
                          (d.__module__ != EState.__name__) and 
                          (d.__module__ != GeometricalIndex.__name__) and 
                          (d.__module__ != GravitationalIndex.__name__) and 
                          (d.__module__ != MoRSE.__name__) and 
                          (d.__module__ != MolecularDistanceEdge.__name__) and 
                          (d.__module__ != MomentOfInertia.__name__) and 
                          (d.__module__ != PBF.__name__) and 
                          (d.__module__ != TopologicalCharge.__name__)), descs)


data = pd.read_csv('cv.csv')
calc = Calculator(descs)
mols = [Chem.MolFromSmiles(smi) for smi in data['smiles']]
X = calc.pandas(mols)

y =data.iloc[:, 2]

# Initialize K-fold cross-validation
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
rmse_list = []
r2_list = []

# Train and evaluate the random forest regression model with cross-validation

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    print('rmse=', rmse)
    rmse_list.append(rmse)
    r2_list.append(r2)
    
avg_rmse = sum(rmse_list) / n_folds
avg_r2 = sum(r2_list) / n_folds

print("Average RMSE:", avg_rmse)
print("Average R2 score:", avg_r2)
