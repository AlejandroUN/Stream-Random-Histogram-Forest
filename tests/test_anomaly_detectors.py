from capymoa.evaluation import AnomalyDetectionEvaluator
from capymoa.anomaly import (
    HalfSpaceTrees,
    OnlineIsolationForest,
    Autoencoder,
    TreeBasedUnsupervised
)
from capymoa.base import Classifier, AnomalyDetector
from capymoa.base import MOAClassifier
from capymoa.datasets import ElectricityTiny
import pytest
from functools import partial
from typing import Callable, Optional
from capymoa.base import _extract_moa_learner_CLI

from capymoa.stream._stream import Schema


@pytest.mark.parametrize(
    "learner_constructor,auc,cli_string",
    [
        (partial(HalfSpaceTrees, window_size=100, number_of_trees=25, max_depth=15), 0.54, None),
        (partial(OnlineIsolationForest, window_size=100, num_trees=32, max_leaf_samples=32), 0.49, None),
        (partial(Autoencoder, hidden_layer=2, learning_rate=0.5, threshold=0.6), 0.42, None),
        (partial(TreeBasedUnsupervised, num_trees=40, max_height=20, window_size=100, random_seed=1), 0.54, None),
    ],
    ids=[
        "HalfSpaceTrees",
        "OnlineIsolationForest",
        "Autoencoder",
        "TreeBasedUnsupervised",
    ],
)
def test_anomaly_detectors(
    learner_constructor: Callable[[Schema], AnomalyDetector],
    auc: float,
    cli_string: Optional[str],
):
    """Test on tiny is a fast running simple test to check if a learner's
    performance has changed.

    Notice how we use the `partial` function to creates a new function with
    hyperparameters already set. This allows us to use the same test function
    for different learners with different hyperparameters.

    :param learner_constructor: A partially applied constructor for the learner
    :param auc: Expected AUC score
    :param cli_string: Expected CLI string for the learner or None
    """
    stream = ElectricityTiny()
    print('defining stream')
    evaluator = AnomalyDetectionEvaluator(schema=stream.get_schema())
    print('defining evaluator')
    learner: AnomalyDetector = learner_constructor(schema=stream.get_schema())
    print('defining learner')

    while stream.has_more_instances():
        instance = stream.next_instance()
        #score_instance is going to compute anomaly score for a given data instance 
        score = learner.score_instance(instance)
        print(type(score))
        print(score)
        evaluator.update(instance.y_index, score)
        learner.train(instance)

    print('finished training')

    # Check if the AUC score matches the expected value for both evaluator types
    actual_auc = evaluator.auc()
    assert actual_auc == pytest.approx(
        auc, abs=0.01
    ), f"Basic Eval: Expected accuracy of {auc:0.1f} got {actual_auc: 0.01f}"
    

    # Optionally check the CLI string if it was provided
    if isinstance(learner, MOAClassifier) and cli_string is not None:
        cli_str = _extract_moa_learner_CLI(learner).strip("()")
        assert cli_str == cli_string, "CLI does not match expected value"
