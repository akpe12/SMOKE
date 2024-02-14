class confusion_matrix:
    def __init__(self) -> None:
        self.confusion_matrix_ = {}
    
    def __call__(self, pred_y, test_y):
        self._compute_confusion_matrix(pred_y, test_y)
    
    def _compute_confusion_matrix(self, pred_y, test_y):
        # TP, TN, FP, FN 계산
        for pred, label in zip(pred_y, test_y):
            # TP, TN
            if pred == label:
                # TP
                if pred == 1 and label == 1:
                    self.confusion_matrix_["TP"] = self.confusion_matrix_.get("TP", 0) + 1
                # TN
                if pred == 0 and label == 0:
                    self.confusion_matrix_["TN"] = self.confusion_matrix_.get("TN", 0) + 1
                            
            # FP, FN
            else:
                # FP
                if pred == 1 and label == 0:
                    self.confusion_matrix_["FP"] = self.confusion_matrix_.get("FP", 0) + 1
                # FN
                if pred == 0 and label == 1:
                    self.confusion_matrix_["FN"] = self.confusion_matrix_.get("FN", 0) + 1
    
    def _compute_accuracy(self):
        acc = (self.confusion_matrix_.get("TP", 0) + self.confusion_matrix_.get("TN", 0)) / (self.confusion_matrix_.get("TP", 0) + self.confusion_matrix_.get("TN", 0) + self.confusion_matrix_.get("FP", 0) + self.confusion_matrix_.get("FN", 0))
        
        return acc
                    
    def _compute_precision(self):
        precision = (self.confusion_matrix_.get("TP", 0)) / (self.confusion_matrix_.get("TP", 0) + self.confusion_matrix_.get("FP", 0))
        
        return precision
    
    def _compute_recall(self):
        recall = (self.confusion_matrix_.get("TP", 0)) / (self.confusion_matrix_.get("TP", 0) + self.confusion_matrix_.get("FN", 0))
        
        return recall
    
    def _compute_f1(self):
        precision = self._compute_precision()
        recall = self._compute_recall()
    
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    def score(self, metric:str = "accuracy"):
        if metric == "accuracy":
            acc = self._compute_accuracy()
            
            return acc
        
        elif metric == "precision":
            precision = self._compute_precision()
            
            return precision
        
        elif metric == "recall":
            recall = self._compute_recall()
            
            return recall
        
        elif metric == "f1":
            f1 = self._compute_f1()
            
            return f1