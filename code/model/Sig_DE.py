import numpy as np


def my_sigmoid(X, D):
    """
    the conversion operator of the sig-DE algorithm
    for discrete variables, map them to {0,1} by a sigmoid function, equ (10)
    :param X: target vector in DE algorithm, with shape (2*D)
    :param D: number of discrete/continues variables
    :return: X1 with shape (2*D)
    """
    assert X.shape[0] == D * 2

    # init the array to return
    x1 = np.copy(X)

    # operate with the first D elements, equ (10)
    sig = 1 / (1 + np.exp(-X[:D]))
    x1[:D] = (np.random.random(size=sig.shape) <= sig)

    return x1


def cal_ic(r_proposed, r_actual):
    """
    Spearman correlation based on equ(5), (objective func / fitness func)
    :param r_proposed: shape (I) score ranking of stock i by the proposed model at time t
    :param r_actual: shape (I) actual return ranking in the next period t+1
    :return: a floating number indicating normalized similarity
    """
    assert r_proposed.shape == r_actual.shape
    return np.cov(r_proposed, r_actual)[0, 1] / np.sqrt(np.var(r_proposed) * np.var(r_actual))


class SigDE:
    def __init__(self, config, Y, r, silent=False):
        """
        initialization of the Sigmoid-DE algorithm class
        :param config: contains hyper-parameters of Sigmoid-DE algorithm
        :param Y: normalized feature Y, a list of T np arrays with shape (I, D)
        :param r: actual rank, a list of T np arrays with shape (I), here T is one timestamp later than T in param Y
        """
        # print or not
        self.silent = silent

        # params
        self.I = [_.shape[0] for _ in Y]  # a list of numbers of training stocks
        self.T = len(Y)  # number of timestamps
        self.D = Y[0].shape[1]  # D-dimension of features, 2 * D if considering discrete and continues variables
        self.P = config.get('P', 30)  # population of DE algorithm
        self.beta = config.get('beta', 0.6)  # a scaled factor in (7) typically \in (0,1)
        self.Cr = config.get('Cr', 0.5)  # the crossover rate in (8) predefined between 0 and 1
        self.G = config.get('G', 100)  # iteration maximum
        self.tol = config.get('tol', 10 ** (-5))  # applied for the termination strategy

        # normalized feature Y
        self.Y = Y  # a list of T np arrays with shape (I, D)

        # actual rank
        self.r = r  # a list of T np arrays with shape (I)
        assert [_.shape[0] for _ in r] == self.I and len(r) == self.T

        # matrix that used for the DE algorithm, with shape (P, 2*D)
        self.X = None  # the population
        self.V = None  # donor vectors generated by the mutation stage
        self.U = None  # trial vectors generated by the crossover stage

        # current generation
        self.g = 0

        # best performance
        self.max_fitness = float('-inf')  # -f in equ(5)
        self.best_variables = np.zeros(shape=2 * self.D)  # best F and W
        # self.selected_stocks = []  # np array of stock indexes
        self.best_g = 0

    def x_initialization(self):
        """
        initialization stage equ (6)
        randomly generate a population of P
        :return: None
        """

        # initialize x_discrete and x_continuous
        x_discrete = -1 + np.random.random(size=[self.P, self.D]) * 2  # \in (-1, 1)
        x_continuous = -1 + np.random.random(size=[self.P, self.D]) * 2  # \in (-1, 1)

        # concat
        self.X = np.concatenate((x_discrete, x_continuous), axis=1)
        assert self.X.shape[0] == self.P and self.X.shape[1] == self.D * 2

    def mutation(self):
        """
        mutation stage equ (7)
        create donor vector V based on X and beta
        :return: None
        """

        # init V
        self.V = np.zeros_like(self.X)

        for i in range(self.P):

            # random r1, different from i
            r1 = np.random.randint(0, self.P)  # 0, 1,..., P-1
            while r1 == i:
                r1 = np.random.randint(0, self.P)

            # random r2, different from i, r1
            r2 = np.random.randint(0, self.P)
            while r2 == r1 or r2 == i:
                r2 = np.random.randint(0, self.P)

            # random r3, different from i, r1, r2
            r3 = np.random.randint(0, self.P)
            while r3 == r1 or r3 == r2 or r3 == i:
                r3 = np.random.randint(0, self.P)

            # equ (7)
            self.V[i] = self.X[r1, :] + self.beta * (self.X[r2, :] - self.X[r3, :])

    def crossover(self):
        """
        crossover stage equ (8)
        generate trial vector U based on target vector X and its donor vector V
        :return: None
        """
        # init U
        self.U = np.zeros_like(self.X)

        # r_d in equ (8)
        rd = np.random.randint(0, self.D * 2)

        # for each column
        for d in range(self.D * 2):
            if d == rd:
                self.U[:, d] = self.V[:, d]
            else:
                mask = (np.random.random(size=self.P) <= self.Cr)
                self.U[:, d] = mask * self.V[:, d] + (~mask) * self.X[:, d]

    def selection(self):
        """
        selection stage with special conversion operator
        update X_{g+1} based on fitness function and X_g U_g
        :return: None
        """

        # selection operator equ (9), fist get a mask
        mask = (
                np.apply_along_axis(func1d=self.fitness_func, axis=1, arr=self.U) <= np.apply_along_axis(
                    func1d=self.fitness_func, axis=1, arr=self.X)
        )  # for each row, call fitness function

        assert mask.shape[0] == self.P

        # apply the mask to X and U
        mask = np.expand_dims(mask, 1).repeat(self.X.shape[1], axis=1)
        X1 = mask * self.U + (~mask) * self.X

        self.X = X1

    def final_score(self, F, W):
        """
        calculate final score used for ranking, equ (3)
        Y with shape (stocks, D, t)
        :param F: discrete variables with shape (D)
        :param W: continues variables with shape (D)
        :return: a list of T np arrays with shape I
        """
        assert F.shape[0] == self.Y[0].shape[1] and W.shape[0] == F.shape[0]

        S = []
        for y in self.Y:
            # y with shape (I,D)
            s = np.einsum('j,ij->i', F * W, y)  # equ (3)
            S.append(s)
        return S

    def fitness_func(self, X):
        """
        fitness function of DE algorithm with a conversion operator specially designed for discrete variables
        :param X: all decision variables with shape (2*D)
        :return: the final_score S with shape (I)
        """
        assert X.shape[0] == self.D * 2

        # conversion operator equ (10)
        x_sig = my_sigmoid(X, self.D)  # with fist D elements \in {0,1} and right D elements \in R

        #  It is worth noticing that if feature j is not selected in solution p at iteration g,
        #  i.e., F_{p,j,g}=0, the corresponding weight W_{p,j,g} is accordingly set to 0.
        mask = (x_sig[:self.D] != 0)
        x_sig[self.D:] = x_sig[self.D:] * mask

        # calculate S_{i,t}, S is a list of T np arrays with shape I
        S = self.final_score(x_sig[:self.D], x_sig[self.D:])  # equ (3)

        # get the rank r_{i,t} of score S_{i,t}, r wis a list of T np arrays with shape I
        r = []
        for index, s in enumerate(S):
            sorted_index = np.argsort(-s, axis=0)  # a list of index(desc)
            r_i = np.zeros_like(sorted_index)  # index list to rank list
            r_i[sorted_index] = np.arange(1, self.I[index] + 1)
            r.append(r_i)
        # r: each column contains (1,2,3,...,)

        # calculate ic_t
        ic_t_list = np.zeros(shape=[self.T])
        for t in range(self.T):
            ic_t_list[t] = cal_ic(r[t], self.r[t])

        # the avg
        fitness = ic_t_list.mean()

        # update best performance
        self.update(x_sig, fitness)

        return fitness

    def update(self, x_sig, fitness):
        """
        update best performance
        :param x_sig: parameters
        :param fitness: fitness value
        :return:
        """
        if fitness > self.max_fitness:
            if not self.silent:
                print('update result! fitness:%f' % fitness)
            self.max_fitness = fitness
            self.best_variables = x_sig
            self.best_g = self.g

    def run(self):
        """
        main function to run the model
        :return: self.best_variables with shape (2*D), containing F and W
        """
        self.x_initialization()
        for g in range(self.G):
            self.g = g
            # termination on condition (2)
            if self.g - self.best_g > 25:
                break
            if g % 20 == 0:
                if not self.silent:
                    print("begin generation %d" % g)
            self.mutation()
            self.crossover()
            self.selection()

        return self.best_variables

