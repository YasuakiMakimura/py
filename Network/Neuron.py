import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    def __init__(self, fw_wscale=None, rec_wscale=None, fb_wscale=None
                 ,*, activation_func):
        # self._output = None
        # self._internal_state = 0
        self._forward_w = fw_wscale
        self._recurrent_w = rec_wscale
        self._feedback_w = fb_wscale
        self._forward_input = None
        self._recurrent_input = None
        self._feedback_input = None
        self._bias = None
        self._gain = None
        self.activation_func = activation_func

    @property
    def gain(self):
        return self._gain

    @property
    def bias(self):
        return self._bias

    @property
    def forward_input(self):
        return self._forward_input

    @property
    def recurrent_input(self):
        return self._recurrent_input

    @property
    def feedback_input(self):
        return self._feedback_input

    @property
    def forward_weight(self):
        return self._forward_w

    @property
    def recurrent_weight(self):
        return self._recurrent_w

    @property
    def feedback_weight(self):
        return self._feedback_w

    @property
    def internal_state(self):
        internal_state = 0
        if self.forward_weight is None:
            internal_state = self.forward_weight.dot(self.forward_input)
        if self.recurrent_weight is None:
            internal_state += self.recurrent_weight.dot(self.recurrent_input)
        if self.feedback_weight is None:
            internal_state += self.feedback_weight.dot(self.feedback_input)
        return internal_state

    @property
    def output(self):
        output = self.activation_func(self.internal_state, self.gain, self.bias)
        return output

    @gain.setter
    def gain(self, gain):
        self._gain = gain

    @bias.setter
    def bias(self, bias):
        self._bias = bias

    @forward_input.setter
    def forward_input(self, input_array: np.ndarray):
        self._forward_input = input_array

    @recurrent_input.setter
    def recurrent_input(self, input_array: np.ndarray):
        self._recurrent_input = input_array

    @feedback_input.setter
    def feedback_input(self, input_array: np.array):
        self._feedback_input = input_array