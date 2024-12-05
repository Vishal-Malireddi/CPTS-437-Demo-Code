import numpy as np

class TreeNode:
    def __init__(self, majClass):
        self.split_feature = -1 # -1 indicates leaf node
        self.children = {} # dictionary of {feature_value: child_tree_node}
        self.majority_class = majClass
        self.overall_majority_class = None

class Forest:
    def __init__(self, num_trees, cont_split, cont_cols):
        self.num_trees = num_trees
        self.trees = []
        self.cont_split = cont_split
        self.cont_cols = cont_cols

# creates a root node and expands down from the root
# arguments:
# -data is the training data
# -depthLimit is the maximum depth of the tree
# -feature_sub is a list of feature indices to use for creating the tree. By default this is empty, indicating all features should be used.
def create_tree(data, depthLimit, feature_sub = []):
  # if the data passed is empty
  if len(data) == 0:
    return None
  
  # calculate class weights
  class_counts = np.unique(data[:,-1], return_counts = True)[1]
  total_class_count = len(data)
  weight_list = total_class_count / class_counts
  weight_dict = {}
  for class_name, class_index in zip(np.unique(data[:,-1]), (range(len(weight_list)))):
    weight_dict[class_name] = weight_list[class_index]

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

  overall_majority = majority_class(data, 0, weight_dict)
  return expand(data, featureDict, depthLimit, 0, overall_majority, weight_dict)

# creates child nodes for a given subset of data. capable of returning leaf nodes if the depth limit is reached or if there is no possible split
# arguments:
# -data is the subset of the training data considered for expanding the tree
def expand(data, featureDict, depthLimit, depth, overall_majority, weight_dict):
  tree_node = TreeNode(majority_class(data, overall_majority, weight_dict))
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
      tree_node.children[value] = expand(child_data, remaining_features, depthLimit, depth + 1, overall_majority, weight_dict)

  return tree_node


# returns the majority class for a subset of data
# arguments:
# -data is the subset of data being examined. it may be the entire training set or a subset of it
def majority_class(data, overall_majority, weight_dict):
  if len(data) == 0:
     return overall_majority
  data = np.asarray(data)
  classes = np.unique(data[:,-1])
  class_counts = np.unique(data[:,-1], return_counts = True)[1]
  weighted_class_counts = class_counts
  for class_name, class_index in zip(classes, range(len(classes))):
    weighted_class_counts[class_index] = class_counts[class_index] * weight_dict[class_name]
  
  return classes[list(weighted_class_counts).index(max(weighted_class_counts))]

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


# returns the predicted class of data_point using a given decision tree
# arguments:
# -tree is the decision tree used to create the prediction
# -data_point is a vector representing a single data point that will be classified
def tree_classify(tree, data_point):
  # if the tree is only the root node, return the majority class
  if tree.split_feature == -1:
    return tree.majority_class

  # select the child node from the data point feature
  #print("trying feature", tree.split_feature, "with data_point[feature] =", data_point[tree.split_feature])
  #print("tree has children:", tree.children.keys())
  if data_point[tree.split_feature] in tree.children:
    child = tree.children[data_point[tree.split_feature]]
  else:
    rand_child = np.random.choice(list(tree.children.keys()))
    child = tree.children[rand_child]

  # if there are more children, keep traversing, otherwise return the majority class
  if child:
    return tree_classify(child, data_point)
  else:
    return tree.majority_class

# returns a Forest object which can be used with forest_classify() to predict a data point's class
# arguments:
# -data is the training set used to train the classifier trees within the forest. The last column of data must be the class feature.
# -depth_limit is the maximum depth of each classifier tree.
# -num_trees is the number of trees to create for the forest.
# -cont_method is the method of handling continuous values. currently supports 'mean' and 'median'
def build_forest(data, depth_limit, num_trees = 5, cont_method = 'mean'):
  # apply the supplied method for handling continuous values
  data, split, cols = apply_cont_method(data, cont_method)

  # check for an invalid method
  if type(data) == type(1):
    print("Invalid continous data handling method")
    return

  # create a base Forest object
  forest = Forest(num_trees, split, cols)

  # for each tree in the forest
  for tree_num in range(num_trees):
    # get a bootstrapped sample
    boot = get_boot_sample(data)
    # get a subset of features
    feature_sub = get_feature_sub(data)
    # build the tree
    forest.trees.append(create_tree(boot, depth_limit, feature_sub))

  return forest


# returns an updated array with an applied method for modifying continuous features. A feature is considered continuous if more than half of its values are unique
# returns 1 if the supplied method is not an existing method
# arguments:
# -data is the array object to update
# -cont_method is the method used to modify continuous features.
def apply_cont_method(data, cont_method):
  # estimate continuous features; a feature is considered continuous if more than half of its values are unique
  cont_cols = []
  # for each column in data
  for col in range(len(data[0])):
     # get a count of unique values
     num_unique = len(np.unique(data[:,col]))
     # compare to the total number of values in the column; if at least half are unique, consider the feature continuous
     if num_unique > len(data[:,col]) / 2:
        cont_cols.append(col)

  # for each continuous feature, apply the supplied method to modify the feature
  split = None
  for col in cont_cols:
    data[:,col] = data[:,col].astype(float)
    # integer divide each feature value by the mean feature value, creating bins
    if cont_method == 'mean':
      split = np.mean(data[:,col])
      data[:,col] = data[:,col] // split

    # integer divide each feature value by the median feature value, creating bins
    elif cont_method == 'median':
      split = np.median(data[:,col])
      data[:,col] = data[:,col] // split
    # the supplied method did not match any existing methods, return 1
    else:
      return 1

  return data, split, cont_cols

# returns a bootstrapped sample from data. The sample is the same size as data, but is randomly sampled with replacement
# arguments:
# -data is the data object from which samples are drawn
def get_boot_sample(data):
  # randomly sample indices from the range of all indicies **with replacement**
  boot_indices = list(np.random.choice(range(len(data)), len(data), replace = True))
  boot = data[boot_indices]
  return boot

# returns a list of m feature indicies from data, where m is the ceiling of the square root of the number of features in data.
# arguments:
# -data is the data object from which features are sampled
def get_feature_sub(data):
  num_features = len(data[0]) - 1
  feature_sub = np.random.choice(range(num_features), int(np.ceil(num_features ** 0.5)) ,replace = False)
  return feature_sub


# takes either a single vector or an array and returns a single predicted class or a list of predicted classes, respectively.
# arguments:
# -forest is the Forest object used to predicted classes
# -data is the object predicted classes are made for. data may either be a single vector, for which a single classification is made, or an array, for which a classification is made for each row.
def forest_classify(forest, data):
  # apply the continuous feature method to the data according to the method tied to the tree
  data = apply_cont_classify_split(data, forest.cont_split, forest.cont_cols)

  # classifying a single point
  if len(data.shape) == 1:
    # create an empty list to hold classifications by each tree
    classifications = []
    # for each tree
    for tree_num in range(len(forest.trees)):
      # predict the class and append the prediction to the classification list
      classifications.append(tree_classify(forest.trees[tree_num], data))
    # take the most frequent prediction. by default, ties are broken by whichever tied element is first in the vector
    majority_prediction = max(classifications, key = classifications.count)

    return majority_prediction

  # classifying a set of points
  else:
    # create an empty list for storing the majority prediction for each row (observation)
    classification_list = []
    # for each row (observation)
    for point in range(len(data)):
      # create an empty list to hold classifications by each tree
      classifications = []
      # for each tree
      for tree_num in range(len(forest.trees)):
        # predict the class for the observation and append the prediction to the list
        classifications.append(tree_classify(forest.trees[tree_num], data[point]))
      # store the most frequent prediction
      majority_prediction = max(classifications, key = classifications.count)
      classification_list.append(majority_prediction)

    return classification_list

def apply_cont_classify_split(data, split, cols):
  # for each continuous feature, apply the supplied method to modify the feature
  for col in cols:
    # integer divide each feature value by the split value, creating bins
    data[:,col] = data[:,col] // split

  return data

def evaluate(predictions, test):
  correct = 0
  for pred_index in range(len(predictions)):
    if predictions[pred_index] == test[pred_index, -1]:
        correct += 1
  return correct / len(predictions)
