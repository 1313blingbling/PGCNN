#核函数的长度尺度，即数据点在距离上的相似性，越小越精细
#较小的nu值会产生更光滑 高斯核函数？
#kernel = Matern(length_scale=1.0, nu=1.5)
import pandas as pd
from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.model_selection import train_test_split, KFold
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mordred import Calculator, PBF, MomentOfInertia, TopologicalCharge, MolecularDistanceEdge, MoRSE, GravitationalIndex, GeometricalIndex, EState, DistanceMatrix, DetourMatrix, CPSA, BaryszMatrix, Autocorrelation, AdjacencyMatrix, descriptors, get_descriptors_from_module


descs = get_descriptors_from_module(descriptors, submodule=True)
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

data = pd.read_csv('cv.csv')#目标变量储存在cv中

# Convert SMILES representation to molecular features对给定的分子进行描述符计算，并将结果存储在一个数据框中
calc = Calculator(descs, ignore_3D=True)
mols = [Chem.MolFromSmiles(smi) for smi in data['smiles']]
X = calc.pandas(mols)#输入特征
y =data.iloc[:, 2]

# Initialize K-fold cross-validation
n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

rmse_list = []
r2_list = []
mae_list = []
# 创建核函数对象并调整超参数
kernel = Matern(length_scale=15, nu=1.5)
# 创建GPR模型对象并调整超参数
gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.5, nugget=0.1, random_state=42)

# 进行交叉验证
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # 在训练集上拟合GPR模型
    gpr.fit(X_train, y_train)
    # 在测试集上进行预测
    y_pred = gpr.predict(X_test)


    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    rmse_list.append(rmse)
    r2_list.append(r2)
    mae_list.append(mae)

avg_rmse = sum(rmse_list) / n_folds
avg_r2 = sum(r2_list) / n_folds
avg_mae = sum(mae_list) / n_folds
# 打印结果
print("Average RMSE:", avg_rmse)
print("Average R2 Score:", avg_r2)
print("Average MAE:", avg_mae)

