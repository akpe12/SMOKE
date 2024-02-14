import numpy as np
from Metric.confusion_matrix import confusion_matrix

class KNeighborsClassifier:
    def __init__(self, n_neighbors:int = 5) -> None:
        self.n_neighbors = n_neighbors
        self._fitted = False
        
        self.X = None
        self.y = None
        
    def set_n_neighbors(self, n_neighbors):
        self.n_neighbors = n_neighbors
        
    def fit(self, X, y):
        # just Storing
        self.X = X
        self.y = y
        
        self._fitted = True
        
    def euclidean_distance(self, data_x, data_y):
        return np.sqrt(((data_x[:, np.newaxis, :] - data_y) ** 2).sum(axis=2))
        
    def predict(self, test_X):
        if self._fitted is not True:
            raise ValueError("The model should be fitted by training dataset before prediction.")
        
        probability = self.predict_proba(test_X)
        
        # predict_proba에서 온 proba로 argmax 해서 making predicted label
        predicted_y = probability.argmax(axis=-1) # (test_X.shape[0], 1)
        
        return predicted_y
        
    def predict_proba(self, test_X):
        if self._fitted is not True:
            raise ValueError("The model should be fitted by training dataset before prediction.")
        
        self.distances = self.euclidean_distance(test_X, self.X)
        
        # 거리 값이 가장 작은 n개의 train 데이터 index를 각 테스트 데이터마다 구하기
        indices = np.argpartition(self.distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]
        
        # 각 테스트 데이터의 train 데이터 index를 train_y와 mapping 해서 label 빼기
        predicted_class = np.empty_like(indices)
        for i in range(predicted_class.shape[0]):
            pred_class = []
            for j in range(predicted_class.shape[1]):
                pred_class.append(self.y[indices[i][j]])
            predicted_class[i] = np.array(pred_class) # (test_X.shape[0], n_neighbors)
        
        # 각 테스트 데이터의 클래스 별 proba 연산
        probability = np.empty((test_X.shape[0], 2)) # (test_X.shape[0], num_classes)
        for i in range(probability.shape[0]):
            _predicted_class = predicted_class[i].tolist()
            neg_prob, pos_prob = _predicted_class.count(0) / self.n_neighbors, _predicted_class.count(1) / self.n_neighbors
            
            probability[i] = np.array([neg_prob, pos_prob])
        
        return probability
    
    def score(self, test_X, test_y, metric:str = "accuracy"):
        predicted_y = self.predict(test_X)
        c_matrix = confusion_matrix()
        c_matrix(predicted_y, test_y)
        
        return c_matrix.score(metric)
    
    def cross_validation(self, val_X, val_y, min_k=1, max_k=99, step=2):
        if (min_k + step) % 2 == 0 or (max_k + step) % 2 == 0:
            raise ValueError(""""min" and "max" values should be odd value, and "step" value should be even.""")
        
        if not self._fitted:
            raise ValueError("The model should be fitted.")
        else:
            best_k = min_k
            best_score = 0
            
            for k in range(min_k, max_k, step):
                self.set_n_neighbors(k)
                
                curr_score = self.score(val_X, val_y, metric="accuracy")
                if curr_score > best_score:
                    best_score = curr_score
                    best_k = k
            
            return best_k, best_score