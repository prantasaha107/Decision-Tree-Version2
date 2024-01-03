from unittest import TestCase
from DecisionTree import*

class TestDecision_Tree(TestCase):
    data = [
        {'label': 'no', 'color': 'blue', 'maker': 'DrLight', 'go_rogue': 'no'},
        {'label': 'yes', 'color': 'red', 'maker': 'DrWily', 'go_rogue': 'yes'},
        {'label': 'no', 'color': 'blue', 'maker': 'DrWily', 'go_rogue': 'no'},
        {'label': 'yes', 'color': 'red', 'maker': 'DrLight', 'go_rogue': 'yes'},
        {'label': 'yes', 'color': 'red', 'maker': 'DrLight', 'go_rogue': 'yes'}
    ]

    def test_get_majority_label(self):
        decision_tree = Decision_Tree()
        majority_label = decision_tree.get_majority_label(self.data)
        self.assertEqual(majority_label, "yes")

    def test_get_best_feature_version2(self):
        decision_tree = Decision_Tree()
        features = {'color': ['blue', 'red'], 'maker': ['DrLight', 'DrWily']}
        best_feature = decision_tree.get_best_feature_version2(self.data, features)
        self.assertEqual(best_feature, 'color')

