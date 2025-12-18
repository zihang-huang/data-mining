import numpy as np
from collections import Counter
from tqdm import tqdm

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
    def __init__(self, loss_function="MSE_mean", leaf_value_estimator="mean", max_depth=20,current_depth=0,min_sample=5,loss_threshold=1e-5, verbose=False, _pbar=None):
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
        self.verbose = verbose
        self._pbar = _pbar
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

        # Initialize progress bar at root level
        if self.verbose and self.current_depth == 0:
            self._pbar = tqdm(total=0, desc="Building RegressionTree", unit="nodes")

        # Update progress bar
        if self._pbar is not None:
            self._pbar.total += 1
            self._pbar.update(0)
            self._pbar.set_postfix(depth=self.current_depth, samples=num_sample)


        # If the number of remaining features = 0 or the classification has meet the standards, return as a leaf node
        # Only the leaf node has the value
        if self.current_depth >= self.max_depth or num_sample <= self.min_sample or isunique or self.loss_function(y)<self.loss_threshold:
            self.isleaf = True
            self.value = self.leaf_value_estimator(y)
            if self._pbar is not None:
                self._pbar.update(1)
            if self.verbose and self.current_depth == 0 and self._pbar is not None:
                self._pbar.close()
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

        if self._pbar is not None:
            self._pbar.update(1)

        # Recurese and construct the decision tree
        if best_split_id != None:
            self.left = RegressionTree(self.loss_function_name, self.estimator_name, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold, verbose=False, _pbar=self._pbar)
            self.left.fit(best_X_left, best_y_left)
            self.right = RegressionTree(self.loss_function_name, self.estimator_name, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold, verbose=False, _pbar=self._pbar)
            self.right.fit(best_X_right, best_y_right)
            # split info
            self.split_id = best_split_id
            self.split_value = best_split_value
            # prepare for pruning
            self.value = self.leaf_value_estimator(y)
        else:
            self.isleaf = True
            self.value = self.leaf_value_estimator(y)

        # Close progress bar at root level
        if self.verbose and self.current_depth == 0 and self._pbar is not None:
            self._pbar.close()
        return self

    # Predict the value given a new instance
    def _predict_single(self, X_new):
        # Only leaf node has the value
        if self.isleaf:
            return self.value
        else:
            if X_new[self.split_id] <= self.split_value:
                return self.left._predict_single(X_new)
            else:
                return self.right._predict_single(X_new)
    
    # Predict the value given a batch of new instances
    def predict(self,X_new):
        if X_new.shape[0] == 1:
            return self._predict_single(X_new)
        else:
            X = np.array(X_new)
            preds = [self._predict_single(x) for x in X]
            return np.array(preds)

    # prune by validation test, from leaf to rootzheg
    def prune(self,X_val, y_val, loss_improvement_threshold=0.0):
        self._prune_calculator(X_val ,y_val, loss_improvement_threshold)

    def _prune_calculator(self, X_val, y_val, loss_improvement_threshold=0.0) -> float:
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)

        # 1) no validation coverage -> prune conservatively (simplify model)
        if y_val.size == 0:
            self.isleaf = True
            self.left = None
            self.right = None
            self.split_id = None
            self.split_value = None
            return 0.0

        # SSE if this node is forced to be a leaf
        loss_leaf = np.sum((y_val - float(self.value)) ** 2)

        # already a leaf (or malformed internal node)
        if self.isleaf or self.split_id is None or self.left is None or self.right is None:
            self.isleaf = True
            self.left = None
            self.right = None
            self.split_id = None
            self.split_value = None
            return loss_leaf

        # split validation data
        mask = X_val[:, self.split_id] <= self.split_value
        X_left, y_left = X_val[mask], y_val[mask]
        X_right, y_right = X_val[~mask], y_val[~mask]

        # subtree loss (after children may prune themselves)
        loss_subtree = self.left._prune_calculator(X_left, y_left, loss_improvement_threshold) \
                    + self.right._prune_calculator(X_right, y_right, loss_improvement_threshold)

        # 2) keep subtree only if it improves by at least threshold
        # improvement = loss_leaf - loss_subtree
        if loss_subtree + loss_improvement_threshold < loss_leaf:
            self.isleaf = False
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
from tqdm import tqdm

class ClassificationTree():
    loss_function_dict = {
        "Gini": Gini,
        "Entropy":Entropy
    }
    estimator_dict = {
        "most_common_vote":most_common_vote
    }

    # initialize a tree
    def __init__(self, loss_function="Entropy", leaf_value_estimator="most_common_vote", max_depth=20,current_depth=0,min_sample=5,loss_threshold=1e-5, verbose=False, _pbar=None):
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
        self.verbose = verbose
        self._pbar = _pbar
        # tree structure
        self.split_id = None
        self.split_value = None
        self.isleaf = None
        self.left = None
        self.right = None
        self.value = None 

    
    # 2 classes accelerator
    def _impurity_binary_vec(self, n: np.ndarray, n1: np.ndarray) -> np.ndarray:
        n = n.astype(np.float64)
        n1 = n1.astype(np.float64) # label = 1
        p = n1 / n

        if self.loss_function_name == "Gini":
            # Gini = 2p(1-p) for binary
            return 2.0 * p * (1.0 - p)

        # Entropy
        with np.errstate(divide="ignore", invalid="ignore"):
            ent = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
        # p=0 or 1 -> entropy should be 0, and above gives nan/inf
        ent[~np.isfinite(ent)] = 0.0
        return ent


    def _best_split_node_binary(self, X: np.ndarray, y: np.ndarray):
        """
        Find best (feature, threshold, loss) for binary y in {0,1}
        using sort + prefix sums. Complexity per feature: O(n log n) for sort + O(n) scan (vectorized).
        """
        n, d = X.shape
        best_loss = float("inf")
        best_f = None
        best_thr = None

        y = y.astype(np.int8, copy=False)

        for f in range(d):
            order = np.argsort(X[:, f], kind="mergesort")
            x_sorted = X[order, f]
            y_sorted = y[order]

            # candidate splits are between i and i+1, i = 0..n-2
            # prefix ones
            ones_prefix = np.cumsum(y_sorted, dtype=np.int32) 
            # ones_prefix[i]为以上有多少label = 1的
            # all posible split position
            left_n = np.arange(1, n, dtype=np.int32) # number of the left subtree         
            left_ones = ones_prefix[:-1]                      
            right_n = n - left_n
            total_ones = ones_prefix[-1]
            right_ones = total_ones - left_ones

            # valid split: feature value changes + leaf size constraints
            valid = (x_sorted[:-1] != x_sorted[1:]) & (left_n >= self.min_sample) & (right_n >= self.min_sample)
            if not np.any(valid):
                continue

            left_imp = self._impurity_binary_vec(left_n, left_ones)
            right_imp = self._impurity_binary_vec(right_n, right_ones)
            loss = (left_n / n) * left_imp + (right_n / n) * right_imp

            # mask invalid splits to +inf
            loss = np.where(valid, loss, np.inf)
            idx = int(np.argmin(loss))
            cur_loss = float(loss[idx]) # min loss under current feature

            if cur_loss < best_loss:
                best_loss = cur_loss
                best_f = f
                # use midpoint threshold
                best_thr = (x_sorted[idx] + x_sorted[idx + 1]) / 2.0

        return best_f, best_thr, best_loss

    # Choose the feature: run if the number of remaining features > 0 and the classification has not meet the standards
    def fit(self, X, y):
        classes, y_int = np.unique(y, return_inverse=True)#转为label=0/1
        n_classes = len(classes)
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        num_sample, num_feature = X.shape
        isunique = (len(np.unique(y)) == 1)

        # Initialize progress bar at root level
        if self.verbose and self.current_depth == 0:
            self._pbar = tqdm(total=0, desc="Building ClassificationTree", unit="nodes")

        # Update progress bar
        if self._pbar is not None:
            self._pbar.total += 1
            self._pbar.update(0)
            self._pbar.set_postfix(depth=self.current_depth, samples=num_sample)

        # If the number of remaining features = 0 or the classification has meet the standards, return as a leaf node
        # Only the leaf node has the value
        if self.current_depth >= self.max_depth or num_sample <= self.min_sample or isunique or self.loss_function(y)<self.loss_threshold:
            self.isleaf = True
            self.value = self.leaf_value_estimator(y)
            if self._pbar is not None:
                self._pbar.update(1)
            if self.verbose and self.current_depth == 0 and self._pbar is not None:
                self._pbar.close()
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

        if n_classes == 2: # accelerator
            best_split_id, best_split_value, best_loss = self._best_split_node_binary(X, y)

            if best_split_id is None:
                self.isleaf = True
                self.value = self.leaf_value_estimator(y)
                return self

            mask = X[:, best_split_id] <= best_split_value
            best_X_left, best_y_left = X[mask], y[mask]
            best_X_right, best_y_right = X[~mask], y[~mask]
        else: # traditional 
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

        if self._pbar is not None:
            self._pbar.update(1)

        # Recurese and construct the decision tree
        if best_split_id != None:
            self.left = ClassificationTree(self.loss_function_name, self.estimator_name, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold, verbose=False, _pbar=self._pbar)
            self.left.fit(best_X_left, best_y_left)
            self.right = ClassificationTree(self.loss_function_name, self.estimator_name, self.max_depth, current_depth=self.current_depth + 1, min_sample=self.min_sample, loss_threshold=self.loss_threshold, verbose=False, _pbar=self._pbar)
            self.right.fit(best_X_right, best_y_right)
            # split info
            self.split_id = best_split_id
            self.split_value = best_split_value
            # prepare for pruning
            self.value = self.leaf_value_estimator(y)
        else:
            self.isleaf = True
            self.value = self.leaf_value_estimator(y)

        # Close progress bar at root level
        if self.verbose and self.current_depth == 0 and self._pbar is not None:
            self._pbar.close()
        return self
        # Check the node infomation

    def node_info(self):
        return {
            "depth": self.current_depth,
            "is_leaf": self.isleaf,
            "split_id": self.split_id,
            "split_value": self.split_value,
            "value": self.value,
        }

    # Predict the value given a new instance
    def _predict_single(self, X_new):
        # Only leaf node has the value
        if self.isleaf:
            return self.value
        else:
            if X_new[self.split_id] <= self.split_value:
                return self.left._predict_single(X_new)
            else:
                return self.right._predict_single(X_new)
    
    # Predict the value given a batch of new instances
    def predict(self,X_new):
        if X_new.shape[0] == 1:
            return self._predict_single(X_new)
        else:
            X = np.array(X_new)
            preds = [self._predict_single(x) for x in X]
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