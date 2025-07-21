
class FNN_:
    def __init__(self, nx, ny, hidden_layers=(2, 1)):
        self.nx = nx
        self.ny = ny
        self.hidden_layers = hidden_layers.copy()
        self.sizes = [nx] + self.hidden_layers + [ny]
        self.W, self.B = self.init_W_B(self.sizes)

    @staticmethod
    def init_W_B(sizes):
        W, B = {}, {}
        for i in range(1, len(sizes)):
            W[i] = np.random.randn(sizes[i - 1], sizes[i])
            B[i] = np.zeros((1, sizes[i]))
        return W, B

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def softmax(x):
        denominator = np.sum(np.exp(x))
        return x / (1.0 * denominator)

    @staticmethod
    def forward_pass(W, B, x, num_neuron_layers):
        H, A = {}, {}
        H[0] = x.flatten()
        for i in range(1, num_neuron_layers):
            A[i] = np.matmul(H[i - 1], W[i]) + B[i]
            H[i] = FNN.sigmoid(A[i])
            H[i] = FNN.sigmoid(A[i]) if i != num_neuron_layers else FNN.softmax(A[i])
        return H[num_neuron_layers]

