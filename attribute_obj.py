import numpy as np


class NumericalAttribute(object):
    def __init__(self, sem_label, num_values):
        self.values = np.array(num_values)
        self.label = sem_label

    def __repr__(self):
        if self.label:
            return self.label
        return "unknown"

    def append_values(self, new_values):
        self.values = np.append(self.values, new_values)


