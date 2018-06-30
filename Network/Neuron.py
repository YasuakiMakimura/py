import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    _w = None
    _u = None
    _o = None

    def __init__(self, n_in, num):
        self.n_in = n_in
        self.w = np.zeros(n_in)
        self.u = np.zeros(num)
        self.o = np.zeros(num)

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, w_array):
        self._w = w_array

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, u_array):
        self._u = u_array

    @property
    def o(self):
        return self._o

    @o.setter
    def o(self, o_array):
        self._o = o_array


# class Layer:
#     _w = None
#     _u = None
#     _o = None
#
#     def __init__(self, *neuron):
#         self.n_in = n_in
#         self.num_in = num
#         self.w = np.zeros((n_in, num))
#         self.u = np.zeros((num,))
#         self.o = np.zeros((num,))
#
#     def __call__(self, input):
#         return np.dot(self.w, input)
#
#     @property
#     def w(self):
#         return self._w
#
#     @w.setter
#     def w(self, w_array):
#         self._w = w_array
#
#     @property
#     def u(self):
#         return self._u
#
#     @u.setter
#     def u(self, u_array):
#         self._u = u_array
#
#     @property
#     def o(self):
#         return self._o
#
#     @o.setter
#     def o(self, o_array):
#         self._o = o_array


class Network:
    def __init__(self, **layer):
        print(layer)
        for l_name, l_inf in layer.items():
            self.__dict__[l_name] = l_inf
        print(self.__dict__)

    # def forward(self, *layer):
    #     layer


if __name__ == "__main__":
    net = Network(l1=Neuron(2, 3))
    net.b = 0
    print(net.b)
