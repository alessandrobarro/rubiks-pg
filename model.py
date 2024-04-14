import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class PolicyNetwork:
    def __init__(self, state_size, hidden_size, action_size):
        self.weights1 = np.random.randn(state_size, hidden_size) * 0.1
        self.bias1 = np.zeros(hidden_size)
        self.weights2 = np.random.randn(hidden_size, hidden_size) * 0.1
        self.bias2 = np.zeros(hidden_size)
        self.weights3 = np.random.randn(hidden_size, action_size) * 0.1
        self.bias3 = np.zeros(action_size)

    def predict(self, state):
        z1 = state.dot(self.weights1) + self.bias1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.weights2) + self.bias2
        a2 = np.tanh(z2)
        z3 = a2.dot(self.weights3) + self.bias3
        return softmax(z3)

    def get_gradients(self, state, action, advantage):
        z1 = state.dot(self.weights1) + self.bias1
        a1 = np.tanh(z1)
        z2 = a1.dot(self.weights2) + self.bias2
        a2 = np.tanh(z2)
        z3 = a2.dot(self.weights3) + self.bias3
        probs = softmax(z3)

        dlog = -probs
        dlog[action] += 1

        grad_w3 = a2[:, None].dot(dlog[None, :])
        grad_b3 = dlog
        delta3 = dlog.dot(self.weights3.T) * (1 - a2**2)
        grad_w2 = a1[:, None].dot(delta3[None, :])
        grad_b2 = delta3
        delta2 = delta3.dot(self.weights2.T) * (1 - a1**2)
        grad_w1 = state[:, None].dot(delta2[None, :])
        grad_b1 = delta2

        grad_w1 *= advantage
        grad_w2 *= advantage
        grad_b2 *= advantage
        grad_w3 *= advantage
        grad_b3 *= advantage

        return grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3

    def update(self, gradients, learning_rate=1e-2):
        grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3 = gradients
        self.weights1 += learning_rate * grad_w1
        self.bias1 += learning_rate * grad_b1
        self.weights2 += learning_rate * grad_w2
        self.bias2 += learning_rate * grad_b2
        self.weights3 += learning_rate * grad_w3
        self.bias3 += learning_rate * grad_b3

    def save_weights(self, filename="policy_weights.npz"):
        np.savez(filename, weights1=self.weights1, bias1=self.bias1,
                 weights2=self.weights2, bias2=self.bias2,
                 weights3=self.weights3, bias3=self.bias3)

    def load_weights(self, filename="policy_weights.npz"):
        data = np.load(filename)
        self.weights1 = data['weights1']
        self.bias1 = data['bias1']
        self.weights2 = data['weights2']
        self.bias2 = data['bias2']
        self.weights3 = data['weights3']
        self.bias3 = data['bias3']

