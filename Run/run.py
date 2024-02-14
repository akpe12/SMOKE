import numpy as np
from Model.model import KNeighborsClassifier
from Data.data import (SmokeDataset,
                        train_val_test_split,
                        normalizing,
                        split_X_y,
                        )
import pickle
#%%
# data processing

# loading data
file_path = "Data/full_dataset.csv"
raw_data = SmokeDataset.load_data(file_path)
#%%
# showing heatmap, dropping inverse features
# raw_data.show_heatmap()
raw_data = raw_data.drop_inverse_features(raw_data, raw_data.corr, threshold=0.1)
#%%
# checking outlier points
raw_data.show_outlier("Humidity[%]")
#%%
# dropping outliers
outliers = ["Humidity[%]", "Raw H2", "Pressure[hPa]"]
filtered_data = raw_data.remove_outliers(raw_data, outliers)
#%%
# checking result of removing outliers
filtered_data.show_outlier("Humidity[%]")
#%%
# splitting data
train_data, val_data, test_data = train_val_test_split(filtered_data, )

train_X, train_y = split_X_y(train_data)
val_X, val_y = split_X_y(val_data)
test_X, test_y = split_X_y(test_data)
#%%
print(train_X.shape)
print(val_X.shape)
print(test_X.shape)
#%%
print(train_X.columns)
#%%
# normalizing data(zero-mean, unit-variance)
train_dataset, val_dataset, test_dataset = normalizing(train_X, val_X, test_X)
train_y, val_y, test_y = train_y.to_numpy(), val_y.to_numpy(), test_y.to_numpy()

# saving data for test time
np.save("test_X.npy", test_dataset)
np.save("test_y.npy", test_y)
#%%
# training
model = KNeighborsClassifier(n_neighbors=1)
model.fit(train_dataset, train_y)

# best_k, best_score = model.cross_validation(val_dataset, val_y, 1, 101) # 결과적으로 best_k = 1
# print(best_k, best_score)

with open("model.pickle", "wb") as f:
    pickle.dump(model, f)

#%%
# test
from Model.model import KNeighborsClassifier
import numpy as np
import pickle

with open("Run/model.pickle", "rb") as f:
    trained_model = pickle.load(f)
    test_X = np.load("Run/test_X.npy")
    test_y = np.load("Run/test_y.npy")
    
    pred_y = trained_model.predict(test_X)
    score = trained_model.score(test_X, test_y, metric="accuracy")
    
    print(pred_y)
    print(score)