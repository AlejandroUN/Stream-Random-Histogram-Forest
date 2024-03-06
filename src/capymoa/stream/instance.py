from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from functools import cached_property

import numpy as np
import numpy.typing as npt
from moa.core import InstanceExample

from capymoa.type_alias import FeatureVector, Label, LabelIndex, TargetValue

if TYPE_CHECKING:
    from capymoa.stream import Schema


class Instance(ABC):
    @property
    @abstractmethod
    def schema(self) -> "Schema":
        """Returns the schema of the instance and the stream it belongs to."""
        ...

    @property
    @abstractmethod
    def x(self) -> FeatureVector:
        """Returns a feature vector containing float values for the instance."""
        ...

    @property
    @abstractmethod
    def java_instance(self) -> InstanceExample:
        """Returns a representation of the instance in Java for use in MOA. This
        method is for advanced users who want to directly interact with MOA's Java
        API.
        """
        ...


class LabeledInstance(Instance, ABC):
    """An :class:`Instance` with a class label.

    .. doctest:: python

       >>> from capymoa.datasets import ElectricityTiny
       ...
       >>> from capymoa.stream.instance import LabeledInstance
       >>> stream = ElectricityTiny()
       >>> instance: LabeledInstance = stream.next_instance()
       >>> instance.y_label
       '1'

       The label and index are NOT the same. One is a human-readable string
       and the other is a integer representation of the class label.
       >>> instance.y_index
       1
       >>> instance.x
       array([0.      , 0.056443, 0.439155, 0.003467, 0.422915, 0.414912])
    """

    @property
    @abstractmethod
    def y_label(self) -> Label:
        """Returns the class label of the instance as a string."""
        ...

    @property
    @abstractmethod
    def y_index(self) -> LabelIndex:
        """Returns the index of the class. It is useful for classification
        tasks as it provides a numeric representation of the class label, ranging
        from zero to the number of classes.
        """
        ...


class RegressionInstance(Instance, ABC):
    """An :class:`Instance` with a continuous target value.

    ..  doctest:: python

        >>> from capymoa.datasets import Fried
        ...
        >>> from capymoa.stream.instance import RegressionInstance
        >>> stream = Fried()
        >>> instance: RegressionInstance = stream.next_instance()
        >>> instance.y_value
        17.949
        >>> instance.x
        array([0.487, 0.072, 0.004, 0.833, 0.765, 0.6  , 0.132, 0.886, 0.073,
               0.342])
    """ 

    @property
    @abstractmethod
    def y_value(self) -> TargetValue:
        """Returns the target value of the instance."""
        ...


class _JavaInstance(Instance):
    def __init__(self, schema: "Schema", java_instance: InstanceExample):
        self._java_instance = java_instance
        self._schema = schema

    @property
    def schema(self) -> "Schema":
        return self._schema

    # This is cached because it can be called multiple times and is never
    # expected to change. It is not pre-computed, however, because it is
    # possible that the user will never access it.
    @cached_property
    def x(self) -> npt.NDArray[np.double]:
        moa_instance = self.java_instance.getData()
        x_array = np.empty(moa_instance.numInputAttributes())
        for i in range(0, moa_instance.numInputAttributes()):
            x_array[i] = moa_instance.value(i)
        return x_array

    @property
    def java_instance(self) -> InstanceExample:
        return self._java_instance


class _JavaLabeledInstance(_JavaInstance, LabeledInstance):
    @cached_property
    def y_label(self):
        return self.schema.get_value_for_index(self.y_index)

    @cached_property
    def y_index(self):
        return int(self.java_instance.getData().classValue())


class _JavaRegressionInstance(_JavaInstance, RegressionInstance):
    @cached_property
    def y_value(self):
        return self.java_instance.getData().classValue()
