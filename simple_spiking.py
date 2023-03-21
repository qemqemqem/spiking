import numpy as np


class LIFNeuron:
    def __init__(self, tau, thr):
        self.tau = tau  # membrane time constant
        self.thr = thr  # spiking threshold
        self.v = 0  # membrane potential

    def update(self, inp):
        # update the membrane potential based on input and leakage
        dv = (inp - self.v) / self.tau
        self.v += dv

        # check for spiking
        if self.v >= self.thr:
            self.v = 0  # reset the membrane potential
            return 1  # return a spike
        else:
            return 0  # no spike


class ReservoirNetwork:
    def __init__(self, n_input, n_hidden, n_output, p=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # initialize the neurons
        self.input_neurons = [LIFNeuron(1, 1) for _ in range(n_input)]
        self.hidden_neurons = [LIFNeuron(10, 1) for _ in range(n_hidden)]
        self.output_neurons = [LIFNeuron(1, 1) for _ in range(n_output)]

        # initialize the connectivity matrix
        self.W_in = np.random.randn(n_hidden, n_input) * p
        self.W_rec = np.random.randn(n_hidden, n_hidden) * p
        self.W_out = np.random.randn(n_output, n_hidden) * p

    def update(self, input_signal):
        # reset the hidden and output neurons
        for neuron in self.hidden_neurons + self.output_neurons:
            neuron.v = 0

        # update the input neurons
        for i, neuron in enumerate(self.input_neurons):
            spike = neuron.update(input_signal[i])
            for j in range(self.n_hidden):
                self.hidden_neurons[j].v += spike * self.W_in[j, i]

        # update the hidden neurons
        for i, neuron in enumerate(self.hidden_neurons):
            spike = neuron.update(0)
            for j in range(self.n_hidden):
                self.hidden_neurons[j].v += spike * self.W_rec[j, i]
            for j in range(self.n_output):
                self.output_neurons[j].v += spike * self.W_out[j, i]

        # update the output neurons and return the output signal
        output_signal = [neuron.update(0) for neuron in self.output_neurons]
        return output_signal

if __name__ == "__main__":
    # create a reservoir network
    in_size = 5
    net = ReservoirNetwork(n_input=in_size, n_hidden=10, n_output=5)

    # simulate the network for 100 time steps
    for t in range(100):
        # create an input signal
        input_signal = np.random.randn(in_size)

        # update the network
        output_signal = net.update(input_signal)

        # print the output signal
        print(output_signal)
