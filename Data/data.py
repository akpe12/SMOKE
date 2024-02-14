import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SmokeDataset():
    def __init__(self, data) -> None:
        self.data = data
        self.corr = data.corr()
        self.shape = self.data.shape
        
    @classmethod
    def load_data(cls, file_path:str):
        raw_data = pd.read_csv(file_path)
        raw_data = raw_data.drop("Unnamed: 0", axis=1)
        
        return cls(raw_data)
    
    def show_heatmap(self):
        plt.figure(figsize=(self.data.shape[1], self.data.shape[1]))
        sns.heatmap(self.corr, cmap='RdBu', annot=True)
        
    def show_outlier(self, feature:str):
        sns.boxplot(x="Fire Alarm", y=feature, data=self.data)

    @classmethod
    def drop_inverse_features(cls, cls_data, corr, threshold:float = .1):
        correlated_features = corr["Fire Alarm"]
        # 비례 상관관계인 features
        filtered_features = correlated_features[correlated_features > threshold]
        
        for feature in cls_data.data.columns:
            if feature not in filtered_features:
                cls_data.data = cls_data.data.drop(feature, axis=1)
                
        return cls(cls_data.data)

    @classmethod
    def remove_outliers(cls, cls_data, outlier_features:list):
        for feature in outlier_features:
            for i in cls_data.data["Fire Alarm"].unique():
                feature_value = cls_data.data[cls_data.data["Fire Alarm"] == i][feature]
                quantile_25 = np.percentile(feature_value, 25)
                quantile_75 = np.percentile(feature_value, 75)
                
                iqr = quantile_75 - quantile_25
                iqr_weight = iqr * 1.5
                
                lowest_val = quantile_25 - iqr_weight
                highest_val = quantile_75 + iqr_weight
                
                outlier_idx = feature_value[(feature_value < lowest_val) | (feature_value > highest_val)].index
                cls_data.data.drop(outlier_idx, axis=0, inplace=True)
                
        return cls(cls_data.data)

def train_val_test_split(cls_data, seed:int = 42, val_ratio:float = .25, test_ratio:float = .4) -> [pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # data shuffle
    cls_data.data = cls_data.data.sample(frac=1, random_state=seed)

    # train, val 데이터셋 나누기
    val_data, train_data = cls_data.data[: int(cls_data.data.shape[0] * val_ratio)], cls_data.data[int(cls_data.data.shape[0] * val_ratio): ]
    # val, test 데이터셋 나누기
    test_data, val_data = val_data[: int(val_data.shape[0] * test_ratio)], val_data[int(val_data.shape[0] * test_ratio): ]
    
    return train_data, val_data, test_data

def split_X_y(dataset):
    y, X = dataset["Fire Alarm"], dataset.drop("Fire Alarm", axis=1)
    
    return X, y

def normalizing(train_data, val_data, test_data):
    mu = np.mean(train_data, axis=0)
    sigma = np.std(train_data, axis=0)

    # normalizing (zero-mean)
    train_data = (train_data - mu) / sigma
    val_data = (val_data - mu) / sigma
    test_data = (test_data - mu) / sigma
    
    return train_data.to_numpy(), val_data.to_numpy(), test_data.to_numpy()

def PCA(X, num_components):
    mean_X = np.mean(X, axis=0)
    X_normalized = X - mean_X
    
    # 공분산 행렬 계산
    covariance_matrix = np.cov(X_normalized, rowvar=False)
    
    # 공분산 행렬의 eigenvalue와 eigenvector 계산
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # eigenvalue를 기준으로 내림차순으로 정렬
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # 주성분 추출
    components = eigenvectors[:, :num_components]
    
    # 데이터를 주성분으로 투영
    projected_data = np.dot(X_normalized, components)
    
    return projected_data