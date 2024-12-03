import numpy as np

class TreeNode:
    def __init__(self, majClass):
        self.split_feature = -1 # -1 indicates leaf node
        self.children = {} # dictionary of {feature_value: child_tree_node}
        self.majority_class = majClass
        self.overall_majority_class = None

class Forest:
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = []

# creates a root node and expands down from the root
# arguments:
# -data is the training data
# -depthLimit is the maximum depth of the tree
# -feature_sub is a list of feature indices to use for creating the tree. By default this is empty, indicating all features should be used.
def create_tree(data, depthLimit, feature_sub = []):
  # if the data passed is empty
  if len(data) == 0:
    return None
  
  # create a dictionary of possible values for each feature
  # if all features are being used
  if len(feature_sub) == 0:
    featureDict = {}
    for feature_index in range(len(data[0]) - 1):
      featureDict[feature_index] = set([example[feature_index] for example in data])

  # if a subset of features is being used
  else:
    featureDict = {}
    for feature_index in feature_sub:
      featureDict[feature_index] = set([example[feature_index] for example in data])

  overall_majority = majority_class(data, 0)
  return expand(data, featureDict, depthLimit, 0, overall_majority)

# creates child nodes for a given subset of data. capable of returning leaf nodes if the depth limit is reached or if there is no possible split
# arguments:
# -data is the subset of the training data considered for expanding the tree
def expand(data, featureDict, depthLimit, depth, overall_majority):
  tree_node = TreeNode(majority_class(data, overall_majority))
  # if no examples for this node, then return leaf node predicting majority class
  if len(data) == 0:
    return tree_node
  # if examples all have same class, then return leaf node predicting this class
  if same_class(data):
      return tree_node
  # if no more features to split on, then return leaf node predicting majority class
  if not featureDict:
      return tree_node

  # find the best feature
  best_feature = find_best_feature(data, featureDict)
  tree_node.split_feature = best_feature

  # update remaining features
  remaining_features = featureDict.copy()
  remaining_features.pop(best_feature)

  # recursively create child nodes for each data subset for the child
  for value in featureDict[best_feature]:
      child_data = filter_data(data, best_feature, value)
      tree_node.children[value] = expand(child_data, remaining_features, depthLimit, depth + 1, overall_majority)

  return tree_node


# returns the majority class for a subset of data
# arguments:
# -data is the subset of data being examined. it may be the entire training set or a subset of it
def majority_class(data, overall_majority):
  if len(data) == 0:
     return overall_majority
  classes = [row[-1] for row in data]
  return max(set(classes), key=classes.count)

# returns whether the data has only one class
# arguments:
# -data is the subset of data being examined. it may be the entire training set or a subset of it
def same_class(data):
  class_value = [example[-1] for example in data]
  return (len(set(class_value)) == 1)

# returns the subset of data with the given feature value for the given feature
# arguments:
# -data is the training data to subset
# -feature is the feature (column) to check for a value in
# -feature_value is the value to check for when subsetting
def filter_data(data, feature, feature_value):
  return list(filter(lambda data_point: data_point[feature] == feature_value, data))

# returns the feature with the most information gain from a dict of features
# arguments:
# -data is the subset of data being examined. it may be the entire training set or a subset of it
# -featureDict is the dictionary of remaining features
def find_best_feature(data, featureDict):
  # defining variables
  best_feature = None
  best_weighted_entropy = np.Inf

  # for each feature, calculate the gain and compare to see if it is better than the previous best
  for feature in featureDict:
    weighted_entropy = get_weighted_entropy(data, featureDict, feature)
    if weighted_entropy < best_weighted_entropy:
      best_info_gain = weighted_entropy
      best_feature = feature

  return best_feature

# returns the total weighted entropy for a feature on a given data subset
# arguments:
# -data is the subset of data being examined. it may be the entire training set or a subset of it
# -featureDict is the dictionary of features
# -feature is the specific feature to calculate the weighted entropy of
def get_weighted_entropy(data, featureDict, feature):
  weighted_entropy = 0.0

  # for each feature value, calculate the entropy and then weight and add to total weighted entropy
  for value in featureDict[feature]:
    child_features = filter_data(data, feature, value)
    weighted_entropy += (float(len(child_features)) / float(len(data))) * get_entropy(child_features)

  return weighted_entropy

# returns the entropy for a subset of data
# arguments:
# -data is the subset of data being examined. it may be the entire training set or a subset of it
def get_entropy(data):
  # defining variables
  class_values = [example[-1] for example in data]
  class_counts = [class_values.count(class_value) for class_value in set(class_values)]
  class_sum = sum(class_counts)
  entropy = 0.0

  for class_count in class_counts:
    if class_count > 0:
      class_fraction = float(class_count) / class_sum
      entropy += -class_fraction * np.log2(class_fraction)

  return entropy

def tree_classify(tree, data_point):
  # if the tree is only the root node, return the majority class
  if tree.split_feature == -1:
    return tree.majority_class

  # select the child node from the data point feature
  child = tree.children[data_point[tree.split_feature]]

  # if there are more children, keep traversing, otherwise return the majority class
  if child:
    return tree_classify(child, data_point)
  else:
    return tree.majority_class

# when building the forest, there should be an argument the user supplies that determines what to do with continuous columns (i.e., split on median/mean)

# could set max number of features to split on (since it will break if we fail to identify a continous column)

# cont_method is the method of handling continuous values. 
def build_forest(data, depth_limit, num_trees = 5, cont_method = 'mean'):
  # apply the supplied method for handling continuous values 
  data = apply_cont_method(data, cont_method)

  if type(data) == type(1):
    print("Invalid continous data handling method")
    return

  forest = Forest(num_trees)

  # for each tree
  for tree_num in range(num_trees):
    # get a bootstrapped sample
    boot = get_boot_sample(data)
    # get a subset of features
    feature_sub = get_feature_sub(data)
    # build the tree
    forest.trees.append( create_tree(boot, depth_limit, feature_sub))

  return forest

def apply_cont_method(data, cont_method):
  # estimate continuous columns; a column is considered continuous if more than half of its values are unique
  cont_cols = []
  for col in range(len(data[0])):
     num_unique = len(np.unique(data[:,col]))
     if num_unique > len(data[:,col]) / 100:
        cont_cols.append(col)
  
  for col in cont_cols:
    if cont_method == 'mean':
     col_mean = np.mean(data[:,col])
     data[:,col] = data[:,col] // col_mean
     
    elif cont_method == 'median':
     col_median = np.median(data[:,col])
     data[:,col] = data[:,col] // col_median
    else:
     return 1

  return data

def get_boot_sample(data):
  # randomly sample indices from the range of all indicies **with replacement**
  boot_indices = list(np.random.choice(range(len(data)), len(data), replace = True))
  boot = data[boot_indices]
  return boot

def get_feature_sub(data):
  num_features = len(data[0])
  feature_sub = np.random.choice(range(num_features), int(np.ceil(num_features ** 0.5)) ,replace = False)
  return feature_sub


# takes either a single data point or an array and returns a single predicted class or a list of predicted classes, respectively.
# 
def forest_classify(forest, data):
  # classifying a single point
  if len(data.shape) == 1:
    classifications = []
    for tree_num in range(len(forest.trees)):
      classifications.append(tree_classify(forest.trees[tree_num], data))
    majority_prediction = max(classifications, key = classifications.count)

    return majority_prediction
  
  # classifying a set of points
  else:
    classification_list = []
    for point in range(len(data)):
      classifications = []
      for tree_num in range(len(forest.trees)):
        classifications.append(tree_classify(forest.trees[tree_num], data[point]))
      majority_prediction = max(classifications, key = classifications.count)
      classification_list.append(majority_prediction)
    
    return classification_list