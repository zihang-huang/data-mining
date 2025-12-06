import numpy as np
from collections import Counter

class DecisionTree():
    # initialize a tree
    def __init__(self, loss_function, leaf_value_estimator, max_depth=5,current_depth=0,min_sample=5,loss_threshold=1e-5):
        self.loss_function = loss_function # Classification: Gini or Entropy; Regression: MSE
        self.leaf_value_estimator = leaf_value_estimator
        self.max_depth = max_depth
        self.current_depth = current_depth
        self.min_sample = min_sample
        self.loss_threshold = loss_threshold
        # tree structure
        self.split_id = None
        self.split_value = None
        self.isleaf = None
        self.left = None
        self.right = None 
        self.value = None 

    # Choose the feature: run if the number of remaining features > 0 and the classification has not meet the standards
    def fit(self, X, y):
        num_sample, num_feature = X.shape
        isunique = (len(np.unique(y)) == 1)
        # If the number of remaining features = 0 or the classification has meet the standards, return as a leaf node
        # Only the leaf node has the value
        if self.current_depth >= self.max_depth or num_sample <= self.min_sample or isunique or self.loss_function(y)<self.loss_threshold:
            self.isleaf = True
            self.value = self.leaf_value_estimator(y)
            return self
        # Else, split and recurse to left and right sub-trees
        best_loss = self.loss_function(y)
        best_split_id = None
        best_split_position = None
        best_split_value = None
        best_X_left = None
        best_X_right = None
        best_y_left = None
        best_y_right = None
        num_feature = X.shape[1]
        Xy = np.concatenate([X, y], 1)
        for feature_id in range(num_feature):
            # sort by given feature
            Xy_sorted = np.array(sorted(Xy, key=lambda x: x[feature_id])) 
            # choose the best split value of this feature
            for split_position in range(len(Xy_sorted)-1):
                X_left = Xy_sorted[:split_position+1,:-1]
                X_right = Xy_sorted[split_position+1:,:-1]
                y_left = Xy_sorted[:split_position+1,-1]
                y_right = Xy_sorted[split_position+1:,-1]
                # calculate loss
                loss_left = len(y_left)/len(y) * self.loss_function(y_left)
                loss_right = len(y_right)/len(y) * self.loss_function(y_right)
                # update the split position
                if (loss_left + loss_right < best_loss):
                    best_split_id = feature_id
                    best_split_position = split_position
                    best_split_value = Xy_sorted[best_split_position, best_split_id]
                    best_loss = loss_left + loss_right
                    best_X_left = X_left
                    best_X_right = X_right
                    best_y_left = y_left
                    best_y_right = y_right
        # Recurese and construct the decision tree
        if best_split_id != None:
            self.left = DecisionTree(self.loss_function, self.leaf_value_estimator, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold)
            self.left.fit(best_X_left, best_y_left)
            self.right = DecisionTree(self.loss_function, self.leaf_value_estimator, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold)
            self.right.fit(best_X_right, best_y_right)
            
            self.split_id = best_split_id
            self.split_value = best_split_value
        else: 
            self.isleaf = True
            self.value = self.leaf_value_estimator(y)
        return self

    # Predict the label/value given a new instance
    def predict(self, X_new):
        # Only leaf node has the value
        if self.isleaf:
            return self.value
        else:
            if X_new[self.split_id] <= self.split_value:
                return self.left.predict(X_new)
            else:
                return self.right.predict(X_new)
            
