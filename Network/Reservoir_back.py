import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from random import random
from math import sqrt
from collections import deque
import Activation_Function
import Initializer
from Neuron import Neuron

class Input:
    def __init__(self):
        pass

    def forward(self, input_array: np.ndarray):
        """
        フォワード入力のarrayを返す関数
        :param input_array:  フィードバック入力の配列
        :return:
        """
        return np.array(input_array)

    def feedback(self, input_array: np.ndarray):
        """
        フィードバック入力のarrayを返す関数
        :param input_array:  フィードバック入力の配列
        :return:
        """
        return np.array(input_array)

class Weight:
    def __init__(self):
        pass

    def forward(self):
        pass

    def feedback(self):
        pass


class ReservoirNeuron:
    _rw_scale = 1.5
    _tau = 10
    _past_internal_state = np.random.uniform(-1, 1, (1,))

    def __init__(self, fw_wscale, rec_wscale, fb_wscale,
                 activation_func=Activation_Function.tanh):
        self.neuron = Neuron(fw_wscale, rec_wscale, fb_wscale,
                             activation_func=activation_func)


    @property
    def forward_weight(self):
        """
        プロパティ
        :return: リザバーニューロンのフォワード結合重み値
        """
        return self.neuron.forward_weight

    @property
    def recurrent_weight(self):
        """
        プロパティ
        :return: リザバーニューロンのリカレント結合重み値
        """
        return self.rw_scale * self.neuron.recurrent_weight

    @property
    def feedback_weight(self):
        """
        プロパティ
        :return: MLRからリザバーニューロンへのフィードバック結合重み値
        """
        return self.neuron.feedback_weight

    @property
    def forward_input(self):
        """
        プロパティ
        :return: フォワード結合部への入力
        """
        return self.neuron.forward_input

    @property
    def recurrent_input(self):
        """
        プロパティ
        :return: リカレント結合部への入力
        """
        return self.neuron.recurrent_input

    @property
    def feedback_input(self):
        """
        プロパティ
        :return: MLRとのフィードバック結合部への入力
        """
        return self.neuron.feedback_input

    @property
    def past_u(self):
        """
        プロパティ
        :return:過去のリザバーニューロンの内部状態
        """
        return self._past_internal_state

    @property
    def u(self):
        """
        プロパティ
        :return: 現在のリザバーニューロンの内部状態
        """
        old_u = self.past_u
        new_u = self.neuron.internal_state
        leaking_rate = 1 / self.tau
        self.past_u = (1 - leaking_rate) * old_u + leaking_rate * new_u
        return self.past_u

    @property
    def o(self):
        """
        プロパティ
        :return: 現在のリザバーニューロンの出力
        """
        return self.neuron.output

    @property
    def tau(self):
        """
        プロパティ
        :return:リザバーニューロンの時定数
        """
        return self._tau

    @property
    def gain(self):
        """
        プロパティ
        :return: ゲイン
        """
        return self.neuron.gain

    @property
    def bias(self):
        """
        プロパティ
        :return: バイアス
        """
        return self.neuron.bias

    @property
    def rw_scale(self):
        return self._rw_scale


    @forward_input.setter
    def forward_input(self, forward_input):
        self.neuron.forward_input = forward_input

    @recurrent_input.setter
    def recurrent_input(self, recurrent_input):
        self.neuron.recurrent_input = recurrent_input

    @feedback_input.setter
    def feedback_input(self, feedback_input):
        self.neuron.feedback_input = feedback_input

    @past_u.setter
    def past_u(self, past_internal_state):
        """
        セッター
        :param past_internal_state:過去のリザバーニューロンの内部状態
        """
        self._past_internal_state = past_internal_state

    @gain.setter
    def gain(self, gain):
        """
        セッター
        :param gain:リザバーニューロンのゲイン
        """
        self.neuron.gain = gain

    @bias.setter
    def bias(self, bias):
        """
        セッター
        :param bias:リザバーニューロンのバイアス
        """
        self.neuron.bias = bias


class Reservoir:
    _connect_rate = 10
    _neurons = deque([])
    _in_w = None
    # _rec_w = deque([])
    # _fb_w = deque([])
    # _u = deque([])

    def __init__(self, num_in: int, num_neuron: int, num_feedback, in_wscale, fb_wscale):
        self.neuron = \
            ReservoirNeuron(
                fw_wscale=Initializer.uniform(-in_wscale, in_wscale, (num_in, )),
                rec_wscale=
                    Initializer.normal(0, 1 / sqrt(num_neuron * (self._connect_rate / 100)),
                                        (num_neuron, )),
                fb_wscale=Initializer.uniform(-fb_wscale, fb_wscale, (num_feedback,))
            )
        for num in range(num_neuron):
            reservoir_neuron = copy(self.neuron)
            self._neurons.append(reservoir_neuron)
            # self._in_w.append(reservoir_neuron.forward_weight)
            # self._rec_w.append(reservoir_neuron.recurrent_weight)
            # self._fb_w.append(reservoir_neuron.feedback_weight)
            # self._u.append(reservoir_neuron.u)
        self._input = np.array([])

    @property
    def connect_rate(self):
        return self._connect_rate

    @connect_rate.setter
    def connect_rate(self, connect_rate):
        self._connect_rate = connect_rate

    @property
    def neurons(self):
        return self._neurons

    @property
    def in_w(self):
        in_w = [neuron.forward_weight for neuron in self.neurons]
        return np.array(in_w)

    # @property
    # def u(self):
    #     old_u =
    #     new_u =



    @property
    def input(self):
        """
        リザバーへのフォワード入力のゲッター
        """
        return self._input

    @input.setter
    def input(self, input_array: np.ndarray):
        """
        リザバーへのフォワード入力のセッター
        :param input_array: リザバーへのフォワード入力
        :type input_array: np.ndarray
        """
        self._input = input_array

    # def internal_state_forming(self):
    #
    #
    #
    # # todo 結合をスパースにする関数
    # def sparsifying(self):
    #     pass
    #
    # # todo リザバーニューロンの内部状態を形成する
    # def u_forming(self):


if __name__ == '__main__':
    # res_neuron = ReservoirNeuron(Initializer.uniform(-1, 1, (2,)),
    #                              Initializer.normal(0, 1, (2,)),
    #                              Initializer.uniform(-1, 1, (2,)))
    # res_neuron.forward_input = np.array([1, 1])
    # res_neuron.recurrent_input = np.array([1, 1])
    # res_neuron.feedback_input = np.array([1, 1])

    reservoir = Reservoir(1, 10, 1, in_wscale=1.0, fb_wscale=1.0)
    print(reservoir.neurons)
    for i in range(10):
        print(id(reservoir.neurons[i]))
    print(reservoir.in_w)