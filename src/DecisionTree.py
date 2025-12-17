import numpy as np
from collections import Counter

###################################
# loss functions
# loss_function_1: classification, entropy
def Entropy(y):
    label_dic = Counter(y) # {label: counts}
    n = len(y)
    entropy = sum(-x/n*np.log2(x/n) for x in label_dic.values())
    return entropy

# loss_fucntion_2: classification, gini index
def Gini(y):
    label_dic = Counter(y) # {label: counts}
    n = len(y)
    gini = 1 - sum(np.square(x/n) for x in label_dic.values())
    return gini

# loss_function_3: regression, variance(l2_norm estimated by the mean value)

# loss_function_4: regression, variance by median
def Var_median(y):
    median = np.median(y)
    loss = np.mean((y - median)**2)
    return loss

###################################
# value estimator
# Classification Esitimator: most common vote
def most_common_vote(y):
    label_dict = Counter(y)
    most_common_label = label_dict.most_common(1)[0][0]
    return most_common_label

# Regression Estimator: mean or median

###################################
# Regression Tree with post-pruning
class RegressionTree():
    loss_function_dict = {
        "MSE_mean": np.var,
        "MSE_median":Var_median
    }
    estimator_dict = {
        "mean":np.mean,
        "median": np.median
    }

    # initialize a tree
    def __init__(self, loss_function="MSE_mean", leaf_value_estimator="mean", max_depth=20,current_depth=0,min_sample=5,loss_threshold=1e-5):
        self.loss_function_name = loss_function
        self.estimator_name = leaf_value_estimator
        try:
            self.loss_function = self.loss_function_dict[loss_function]
        except KeyError:
            raise ValueError(
            f"Unknown loss_function '{loss_function}'. "
            f"Available: {list(self.loss_function_dict.keys())}"
        )
        try:
            self.leaf_value_estimator = self.estimator_dict[leaf_value_estimator]
        except KeyError:
            raise ValueError(
            f"Unknown leaf_value_estimator '{leaf_value_estimator}'. "
            f"Available: {list(self.estimator_dict.keys())}"
        )
        
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

    # Check the node infomation
    def node_info(self):
        return {
            "depth": self.current_depth,
            "is_leaf": self.isleaf,
            "split_id": self.split_id,
            "split_value": self.split_value,
            "value": self.value,
        }

    # Choose the feature: run if the number of remaining features > 0 and the classification has not meet the standards
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
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
        # y is (n,)
        y_copied = y.reshape(-1,1)
        Xy = np.concatenate([X, y_copied], 1)
        # RegressionTree can use the same feature in different nodes
        for feature_id in range(num_feature):
            # sort by given feature
            Xy_sorted = np.array(sorted(Xy, key=lambda x: x[feature_id])) 
            # choose the best split value of this feature
            for split_position in range(len(Xy_sorted)-1):
                # for the same value, split at the last one
                if Xy_sorted[split_position,feature_id] == Xy_sorted[split_position+1,feature_id]:
                    continue
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
            self.left = RegressionTree(self.loss_function_name, self.estimator_name, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold)
            self.left.fit(best_X_left, best_y_left)
            self.right = RegressionTree(self.loss_function_name, self.estimator_name, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold)
            self.right.fit(best_X_right, best_y_right)
            # split info
            self.split_id = best_split_id
            self.split_value = best_split_value
            # prepare for pruning
            self.value = self.leaf_value_estimator(y)
        else: 
            self.isleaf = True
            self.value = self.leaf_value_estimator(y)
        return self

    # Predict the value given a new instance
    def predict_single(self, X_new):
        # Only leaf node has the value
        if self.isleaf:
            return self.value
        else:
            if X_new[self.split_id] <= self.split_value:
                return self.left.predict_single(X_new)
            else:
                return self.right.predict_single(X_new)
    
    # Predict the value given a batch of new instances
    def predict_batch(self,X_new):
        X = np.array(X_new)
        preds = [self.predict_single(x) for x in X]
        return np.array(preds)

    # prune by validation test, from leaf to rootzheg
    def prune(self,X_val, y_val, loss_improvement_threshold=0.0):
        self._prune_calculator(X_val ,y_val, loss_improvement_threshold)

    def _prune_calculator(self,X_val, y_val,loss_improvement_threshold=0.0) -> float:
        if len(y_val) == 0:
            return 0.0
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        # calculate the loss if this node were leaf node
        preds_leaf = np.full_like(y_val, self.value, dtype=float)
        loss_leaf = np.sum((y_val - preds_leaf)**2)
        if self.isleaf:
            return loss_leaf
        # if not leaf, compare the loss of splitting as before and as a leaf node
        left_id = X_val[:,self.split_id]<= self.split_value
        X_left, y_left = X_val[left_id], y_val[left_id]
        X_right, y_right = X_val[~left_id], y_val[~left_id]

        loss_subtree = 0.0
        if self.left is not None:
            loss_subtree += self.left._prune_calculator(X_left, y_left, loss_improvement_threshold)
        if self.right is not None:
            loss_subtree += self.right._prune_calculator(X_right, y_right, loss_improvement_threshold)
        
        if loss_subtree < loss_leaf + loss_improvement_threshold: 
            # not prune
            return loss_subtree
        else:
            self.isleaf = True
            self.left = None
            self.right = None
            self.split_id = None
            self.split_value = None 
            return loss_leaf

    # check the information of the whole tree
    def _collect_stats(self):
        if self is None:
            return 0, 0, 0, Counter()
        node_count = 1

        if self.isleaf:
            leaf_count = 1
            actual_depth = 1      # 以当前节点为根的子树深度 = 1
            feature_counter = Counter()
            return node_count, leaf_count, actual_depth, feature_counter

        # else, recursively calculate
        left_nodes, left_leaves, left_depth, left_counter = (0, 0, 0, Counter())
        right_nodes, right_leaves, right_depth, right_counter = (0, 0, 0, Counter())

        if self.left is not None:
            left_nodes, left_leaves, left_depth, left_counter = self.left._collect_stats()
        if self.right is not None:
            right_nodes, right_leaves, right_depth, right_counter = self.right._collect_stats()

        node_count += left_nodes + right_nodes
        leaf_count = left_leaves + right_leaves
        actual_depth = 1 + max(left_depth, right_depth)   

        feature_counter = left_counter + right_counter
        if self.split_id is not None:
            feature_counter[self.split_id] += 1

        return node_count, leaf_count, actual_depth, feature_counter

    def get_stats(self, num_features, feature_names):
        n_nodes, n_leaves, actual_depth, feature_counter = self._collect_stats()
        feature_counts_array = np.zeros(num_features, dtype=int)

        for fid, cnt in feature_counter.items():
            if fid is not None and 0 <= fid < num_features:
                feature_counts_array[fid] = cnt

        feature_name_counter = {
        feature_names[fid]: count for fid, count in feature_counter.items()
        }
        return {
            "n_nodes": n_nodes,
            "n_leaves": n_leaves,
            "actual_depth": actual_depth,         # 真实树深度
            "feature_counts": feature_name_counter,      # Counter({feature_id: 次数})
        }

# Classification Tree
# accelerated version!
class ClassificationTree():
    loss_function_dict = {
        "Gini": Gini,
        "Entropy":Entropy
    }
    estimator_dict = {
        "most_common_vote":most_common_vote
    }

    # initialize a tree
    def __init__(self, loss_function="Entropy", leaf_value_estimator="most_common_vote", max_depth=20,current_depth=0,min_sample=5,loss_threshold=1e-5):
        self.loss_function_name = loss_function
        self.estimator_name = leaf_value_estimator
        try:
            self.loss_function = self.loss_function_dict[loss_function]
        except KeyError:
            raise ValueError(
            f"Unknown loss_function '{loss_function}'. "
            f"Available: {list(self.loss_function_dict.keys())}"
        )
        try:
            self.leaf_value_estimator = self.estimator_dict[leaf_value_estimator]
        except KeyError:
            raise ValueError(
            f"Unknown leaf_value_estimator '{leaf_value_estimator}'. "
            f"Available: {list(self.estimator_dict.keys())}"
        )
        
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

    # Check the node infomation
    def node_info(self):
        return {
            "depth": self.current_depth,
            "is_leaf": self.isleaf,
            "split_id": self.split_id,
            "split_value": self.split_value,
            "value": self.value,
        }

    # Choose the feature: run if the number of remaining features > 0 and the classification has not meet the standards
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
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
        # RegressionTree can use the same feature in different nodes
        for feature_id in range(num_feature):
            # sort by given feature
            order = np.argsort(X[:, feature_id])
            X_sorted = X[order]
            y_sorted = y[order]
            # choose the best split value of this feature
            for split_position in range(n-1):
                # for the same value, split at the last one
                if X_sorted[split_position,feature_id] == X_sorted[split_position+1,feature_id]:
                    continue
                X_left = X_sorted[:split_position+1]
                X_right = X_sorted[split_position+1:]
                y_left = y_sorted[:split_position+1]
                y_right = y_sorted[split_position+1:]
                # calculate loss
                loss_left = len(y_left)/len(y) * self.loss_function(y_left)
                loss_right = len(y_right)/len(y) * self.loss_function(y_right)
                # update the split position
                if (loss_left + loss_right < best_loss):
                    best_split_id = feature_id
                    best_split_position = split_position
                    best_split_value = X_sorted[best_split_position, best_split_id]
                    best_loss = loss_left + loss_right
                    best_X_left = X_left
                    best_X_right = X_right
                    best_y_left = y_left
                    best_y_right = y_right
        # Recurese and construct the decision tree
        if best_split_id != None:
            self.left = ClassificationTree(self.loss_function_name, self.estimator_name, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold)
            self.left.fit(best_X_left, best_y_left)
            self.right = ClassificationTree(self.loss_function_name, self.estimator_name, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold)
            self.right.fit(best_X_right, best_y_right)
            # split info
            self.split_id = best_split_id
            self.split_value = best_split_value
            # prepare for pruning
            self.value = self.leaf_value_estimator(y)
        else: 
            self.isleaf = True
            self.value = self.leaf_value_estimator(y)
        return self

    # Predict the value given a new instance
    def predict_single(self, X_new):
        # Only leaf node has the value
        if self.isleaf:
            return self.value
        else:
            if X_new[self.split_id] <= self.split_value:
                return self.left.predict_single(X_new)
            else:
                return self.right.predict_single(X_new)
    
    # Predict the value given a batch of new instances
    def predict_batch(self,X_new):
        X = np.array(X_new)
        preds = [self.predict_single(x) for x in X]
        return np.array(preds)

    # prune by validation test, from leaf to rootzheg
    def prune(self,X_val, y_val, loss_improvement_threshold=0.0):
        self._prune_calculator(X_val ,y_val, loss_improvement_threshold)

    def _prune_calculator(self,X_val, y_val,loss_improvement_threshold=0.0) -> float:
        if len(y_val) == 0:
            return 0.0
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        # calculate the loss if this node were leaf node
        preds_leaf = np.full_like(y_val, self.value, dtype=float)
        loss_leaf = np.sum(preds_leaf != y_val)
        if self.isleaf:
            return loss_leaf
        # if not leaf, compare the loss of splitting as before and as a leaf node
        left_id = X_val[:,self.split_id]<= self.split_value
        X_left, y_left = X_val[left_id], y_val[left_id]
        X_right, y_right = X_val[~left_id], y_val[~left_id]

        loss_subtree = 0.0
        if self.left is not None:
            loss_subtree += self.left._prune_calculator(X_left, y_left, loss_improvement_threshold)
        if self.right is not None:
            loss_subtree += self.right._prune_calculator(X_right, y_right, loss_improvement_threshold)
        
        if loss_subtree < loss_leaf + loss_improvement_threshold: 
            # not prune
            return loss_subtree
        else:
            self.isleaf = True
            self.left = None
            self.right = None
            self.split_id = None
            self.split_value = None 
            return loss_leaf
    # check the information of the whole tree
    def _collect_stats(self):
        if self is None:
            return 0, 0, 0, Counter()
        node_count = 1

        if self.isleaf:
            leaf_count = 1
            actual_depth = 1      # 以当前节点为根的子树深度 = 1
            feature_counter = Counter()
            return node_count, leaf_count, actual_depth, feature_counter

        # else, recursively calculate
        left_nodes, left_leaves, left_depth, left_counter = (0, 0, 0, Counter())
        right_nodes, right_leaves, right_depth, right_counter = (0, 0, 0, Counter())

        if self.left is not None:
            left_nodes, left_leaves, left_depth, left_counter = self.left._collect_stats()
        if self.right is not None:
            right_nodes, right_leaves, right_depth, right_counter = self.right._collect_stats()

        node_count += left_nodes + right_nodes
        leaf_count = left_leaves + right_leaves
        actual_depth = 1 + max(left_depth, right_depth)   

        feature_counter = left_counter + right_counter
        if self.split_id is not None:
            feature_counter[self.split_id] += 1

        return node_count, leaf_count, actual_depth, feature_counter

    def get_stats(self, num_features, feature_names):
        n_nodes, n_leaves, actual_depth, feature_counter = self._collect_stats()
        feature_counts_array = np.zeros(num_features, dtype=int)

        for fid, cnt in feature_counter.items():
            if fid is not None and 0 <= fid < num_features:
                feature_counts_array[fid] = cnt

        feature_name_counter = {
        feature_names[fid]: count for fid, count in feature_counter.items()
        }
        return {
            "n_nodes": n_nodes,
            "n_leaves": n_leaves,
            "actual_depth": actual_depth,         # 真实树深度
            "feature_counts": feature_name_counter,      # Counter({feature_id: 次数})
        }


        
