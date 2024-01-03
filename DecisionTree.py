"""
Decision Tree Classifier

This program implements a basic decision tree for binary classification problems.
It reads data from a file, builds a decision tree based on the features in the data,
and allows classification of new samples.
The tree-building process uses information gain to determine the most informative features for splitting.

"""



import math as math


class Decision_Treenode(object):
    """Base class for decision tree nodes."""

    def __init__(self):
        """Initialize a Decision_Treenode."""
        return

    def classify(self, sample):
        """Return the label for the given sample."""
        pass


class Label_Node(Decision_Treenode):

    def __init__(self, label):
        """Initialize a Label_Node with a given label.

        :param label: The label for the node.
        """
        super().__init__()
        self.label = label

    def classify(self, sample):
        """Return the label associated with this node."""
        return self.label

    def __str__(self):
        """String representation of the Label_Node."""
        if self.label is not None:
            return "Label " + self.label
        else:
            return "Label None"


class Feature_Node(Decision_Treenode):

    def __init__(self, feature, values):
        """Return a new feature node that splits on the given feature.

        :param feature: String, name of the categorical feature this node will split on.
        :param values: List of all possible values for that feature.

        Returns: A new feature node with no children yet.
        """
        self.feature = feature
        self.children = {v: None for v in values}

    def classify(self, sample):
        """Return the label for the given sample by traversing the tree."""
        sample_value = sample[self.feature]
        child_node = self.children[sample_value]
        return child_node.classify(sample)

    def __str__(self):
        """String representation of the Feature_Node."""
        return "Feature " + self.feature


class Decision_Tree(object):

    def __init__(self):
        """Initialize a Decision_Tree."""
        self.root = None
        self.outputname = ""

    def build(self, filename):

        """
                Builds a binary decision tree from data in the given file.

                The file must be plain text with one column for each feature.
                The first column is the sample ID, and the last column is the label.
                Columns are white-space separated, and sample labels must be 'yes' or 'no'.

                :param filename: The name of the file to open.
        """
        try:
            with open(filename, 'r') as f:
                features = f.readline()
                features = features.strip().split()

                # get rid of the sample_id
                features = features[1:]

                # get rid of the label name in the last column
                self.outputname = features[-1]
                features = features[:-1]

                # construct a list of all possible values for each feature,
                # so that we're not limited to binary features
                feature_vals = {}
                for feat in features:
                    feature_vals[feat] = []

                data = []
                for line in f:
                    line = line.strip().split()
                    line = line[1:]  # get rid of the sample_id
                    s = {}
                    s["label"] = line[-1]
                    for i in range(len(features)):
                        s[features[i]] = line[i]
                        if line[i] not in feature_vals[features[i]]:
                            feature_vals[features[i]].append(line[i])

                    data.append(s)

            # store the possible values for features for later queries if needed
            self.features = feature_vals
            majority = self.get_majority_label(data)

            if len(data) > 0:
                self.root = self.__build_rec(data, feature_vals, majority)

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"Error: An unexpected error occurred - {e}")

    def calculate_uncertainty(self, data):

        """
               Returns the entropy of a given data set.

               :param data: A list of records representing the data.
               :return: The entropy (float) of the given data set.
               """
        num_records = len(data)
        if num_records == 0:
            return 0

        num_yes = len([record for record in data if record["label"] == "yes"])
        num_no = num_records - num_yes

        if num_yes == 0 or num_no == 0:
            return 0

        p_yes = num_yes / num_records
        p_no = num_no / num_records
        # Formula for calculating Uncertainty

        uncertainty = - (p_yes * math.log2(p_yes) + p_no * math.log2(p_no))

        return uncertainty

    def get_best_feature_version1(self, data, features):
        """This is just a basic feature and DOESN'T do much as the features are selected in arbitrary manner"""
        return list(features.keys())[0]

    def get_best_feature_version2(self, data, features):

        """
               Returns the name of the feature with the highest information gain.

               :param data: A list of records representing the data.
               :param features: A dictionary mapping a feature name to a list of all possible values for that feature.
               :return: The name (string) of the most informative feature on which to split the given data.
               """
        original_uncertainty = self.calculate_uncertainty(data)
        best_feature = None
        best_information_gain = -1

        for feature in features:
            # information_gain = 0
            attribute_uncertainty = 0
            for value in features[feature]:
                subset = [record for record in data if record[feature] == value]
                weighted_average = len(subset) / len(data)
                attribute_uncertainty += weighted_average * self.calculate_uncertainty(subset)

            information_gain = original_uncertainty - attribute_uncertainty

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature = feature

        return best_feature

    def get_majority_label(self, data):
        """
              Returns the majority label from the data set. If there is a tie, an arbitrary label is returned.

              :param data: A list of records representing the data.
              :return: The majority label (string) from the data set.
              """
        histo = {}
        result = ""
        for d in data:
            label = d["label"]
            result = label
            if label not in histo:
                histo[label] = 1
            else:
                histo[label] += 1

        for label in histo:
            if histo[label] > histo[result]:
                result = label
        return result

    def __build_rec(self, data, features, mostcommon):
        """Recursively build a decision tree with the given data and features, returning the root of that tree.

        :param data: A list of records representing the data. Each record MUST have a field called "label",
                     and other fields must be the name of a feature.
        :param features: Dictionary mapping a feature name to a list of all possible values for that feature.
        :param mostcommon: String, most common label of the parent node. This is needed in the event
                          we find a feature combination of which we have seen no examples.
        """
        # base case 1: data is empty, meaning we don't have any point with the current feature values
        # use the most common label from the parent's data set
        if len(data) == 0:
            return Label_Node(mostcommon)
        # base case 2: features are empty, so return label node with majority class
        elif len(features.keys()) == 0:
            majority = self.get_majority_label(data)
            return Label_Node(majority)
        else:

            # base case 3: all the remaining data has the same label, so features don't matter
            all_same = True
            first = data[0]["label"]
            for d in data:
                if d["label"] != first:
                    all_same = False
                    break

            if all_same:
                return Label_Node(first)
            else:
                # recursive case: split on some feature

                # get the majority label to pass down in case there are child nodes with no data
                majority = self.get_majority_label(data)

                # get the feature on which to split
                feat = self.get_best_feature_version2(data, features)

                feat_vals = features.pop(feat)
                node = Feature_Node(feat, feat_vals)

                # for each possible value of the chosen feature, partition the data based on that value
                for v in feat_vals:
                    partition = []
                    for d in data:
                        if d[feat] == v:
                            partition.append(d)

                    child = self.__build_rec(partition, features, majority)
                    node.children[v] = child

                # make sure to add the feature back!
                features[feat] = feat_vals
                return node

    def classify(self, sample):
        """Return the classification label of the given sample.

        :param sample: A record representing a data point. Keys of the record must be
                       feature names that are in the tree.

        Returns: The label (string) for the given sample.
        """
        return self.root.classify(sample)

    def size(self):
        """Return the total number of nodes of any kind in this decision tree."""
        return self.__size_rec(self.root)

    def __size_rec(self, node):
        """Helper method to recursively calculate the size of the decision tree."""
        if node is None:
            return 0
        elif isinstance(node, Label_Node):
            return 1
        else:
            numchildren = 0
            for child in node.children.values():
                numchildren += self.__size_rec(child)

            return 1 + numchildren

    def __str__(self):
        """String representation of the Decision_Tree."""
        result = "Decision tree to determine the " + self.outputname + " of a data point.\n******\n"
        # do a breadth-first traversal of the tree to print.
        queue = []
        # keep track of the depth of each node as we put it into the queue
        root_node_info = ("Root", self.root, 0)
        queue.append(root_node_info)
        cur_depth = 0
        layer = []

        while len(queue) > 0:
            current_node_info = queue.pop(0)
            parent_value = current_node_info[0]
            node = current_node_info[1]
            depth = current_node_info[2]
            if depth != cur_depth:
                cur_depth += 1
                result += " |--| ".join(layer)
                result += "\n\nNodes at layer " + str(cur_depth) + ": "
                layer = []
            layer.append(parent_value + " " + str(node))
            if isinstance(node, Feature_Node):
                for value, child_node in node.children.items():
                    next_node_info = (node.feature + "=" + value, child_node, cur_depth + 1)
                    queue.append(next_node_info)

        result += " |--| ".join(layer)
        return result

