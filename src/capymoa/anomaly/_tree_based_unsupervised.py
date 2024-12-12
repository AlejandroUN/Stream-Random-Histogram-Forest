from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
from capymoa.type_alias import AnomalyScore
import numpy as np
import random

class TreeBasedUnsupervised(AnomalyDetector):
    """Tree-Based Unsupervised Anomaly Detector based on STREAMRHF."""

    def __init__(self, schema=None, num_trees=40, max_height=20, window_size=100, random_seed=1):
        super().__init__(schema, random_seed=random_seed)
        self.num_trees = num_trees
        self.max_height = max_height
        self.window_size = window_size
        self.random_seed = random_seed
        self.forest = []
        self.reference_window = []
        self.current_window = []

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

        self._initialize_forest()

    def _initialize_forest(self):
        """Initialize a forest of random histogram trees."""
        for _ in range(self.num_trees):
            tree = self._create_tree()
            self.forest.append(tree)

    def _create_tree(self):
        """Create an empty tree structure."""
        return {
            "splits": {},  # Node split information
            "leaves": {},  # Leaf statistics
        }

    def _split_node(self, data, node_id, depth):
        """Split a node based on kurtosis."""
        if depth >= self.max_height or len(data) <= 1:
            # Leaf node, return with data
            return {"leaf": True, "data": data}

        # Calculate kurtosis for each attribute
        kurtosis_values = [self._calculate_kurtosis(data[:, i]) for i in range(data.shape[1])]
        total_kurtosis = sum(np.log1p(kurtosis_values))

        # Select splitting attribute and value
        r = random.uniform(0, total_kurtosis)
        split_attr = np.searchsorted(np.cumsum(np.log1p(kurtosis_values)), r)
        split_val = random.uniform(np.min(data[:, split_attr]), np.max(data[:, split_attr]))

        # Split data
        left_data = data[data[:, split_attr] <= split_val]
        right_data = data[data[:, split_attr] > split_val]

        # Return a non-leaf node with the necessary attributes
        return {
            "leaf": False,
            "attribute": split_attr,  # The attribute used for splitting
            "value": split_val,       # The value for the split
            "left": self._split_node(left_data, node_id * 2 + 1, depth + 1),  # Left child node
            "right": self._split_node(right_data, node_id * 2 + 2, depth + 1), # Right child node
        }


    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of a dataset."""
        mean = np.mean(data)
        variance = np.var(data)
        fourth_moment = np.mean((data - mean) ** 4)
        return fourth_moment / (variance ** 2) if variance > 0 else 0

    def train(self, instance: Instance):
        """Train the model with a new instance."""
        self.current_window.append(instance.x)
        if len(self.current_window) >= self.window_size:
            # Update reference window and rebuild the forest
            self.reference_window = self.current_window.copy()
            self.current_window = []
            self.forest = []
            self._initialize_forest()

    def score_instance(self, instance: Instance) -> AnomalyScore:
        """Calculate the anomaly score for an instance."""
        scores = []
        for tree in self.forest:
            leaf_size = self._get_leaf_size(tree, instance.x)
            if leaf_size > 0:
                scores.append(np.log(1 / leaf_size))
        return sum(scores)

    def _get_leaf_size(self, tree, instance):
        """Traverse the tree to find the leaf size for a given instance."""
        node = tree
        while not node.get("leaf", False):  # Check if it's not a leaf
            if "attribute" not in node:
                # This case should not happen, but if it's encountered, the tree is malformed.
                return 0
            attr = node["attribute"]
            if instance[attr] <= node["value"]:
                node = node["left"]
            else:
                node = node["right"]

        # Once a leaf is reached, return the size of the data in the leaf
        return len(node["data"]) if "data" in node else 0


    def predict(self, instance: Instance) -> int:
        """Predict whether an instance is normal (0) or anomalous (1)."""
        anomaly_score = self.score_instance(instance)
        
        # Define a threshold for anomaly detection, e.g., 1.0
        threshold = 0.5  # Adjust this threshold based on your requirements
        if anomaly_score > threshold:
            return 1  # Anomaly
        else:
            return 0  # Normal

