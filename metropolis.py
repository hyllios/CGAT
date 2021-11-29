import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


class MarkovChain:

    def __init__(self, distribution, generator, start=None, *args, **kwargs):
        self.distribution = distribution
        self.generator = generator
        self.args = args
        self.kwargs = kwargs
        self.chain = []
        if start is None:
            x = generator(*args, **kwargs)
            p = distribution(x)
            while p <= 0:
                x = generator(*args, **kwargs)
                p = distribution(x)
            self.chain.append(x)
        else:
            self.chain.append(start)

    def __getitem__(self, item):
        return self.chain[item]

    def __iter__(self):
        return self.chain.__iter__()

    def __len__(self):
        return self.chain.__len__()

    def step(self, n: int = 1):
        for _ in range(n):
            y = self.generator(*self.args, **self.kwargs)
            p = min(1, self.distribution(y) / self.distribution(self[-1]))
            if random.random() <= p:
                self.chain.append(y)
            else:
                self.chain.append(self[-1])


def main():
    def distribution(x):
        return 1 if np.abs(x) > .5 else 0

    def distribution2(skew):
        def f(x):
            return 2 / np.sqrt(2 * np.pi) * np.exp(-np.square(x) / 2) * (1 + erf(skew * x / np.sqrt(2))) / 2

        return f

    def generator():
        return random.random() * 6 - 3

    def normal_distribution(x):
        return np.exp(-np.square(x) / 2) / np.sqrt(2 * np.pi)

    def cum_distribution(x):
        return (1 + erf(x / np.sqrt(2))) / 2

    def get_skewd_gaussion(skew, location, scale):
        def f(x):
            return 2 / scale * normal_distribution((x - location) / scale) * cum_distribution(
                skew * (x - location) / scale)

        return f

    chain = MarkovChain(get_skewd_gaussion(100, -0.02, 0.7), lambda: random.random() * 3.5)
    n = 50_000
    chain.step(n)
    plt.hist(chain, weights=[1 / len(chain)] * len(chain), bins=100)
    plt.show()
    #
    # x = np.linspace(-3, 3, 600)
    # skew = 4
    # y = 2 / np.sqrt(2 * np.pi) * np.exp(-np.square(x) / 2) * (1 + erf(skew * x / np.sqrt(2))) / 2
    # plt.plot(x, y)
    # plt.show()


if __name__ == '__main__':
    main()
