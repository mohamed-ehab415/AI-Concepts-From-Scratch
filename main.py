import numpy as np
from numpy.linalg import norm

def gradient_descent(fderiv, inital_start, step_size = 0.001, precision = 0.00001, max_iter = 1000):
    cur_start = np.array(inital_start)
    last_start = cur_start + 100 * precision    # something different
    start_list = [cur_start]

    iter = 0
    while norm(cur_start - last_start) > precision and iter < max_iter:
        print(cur_start)
        last_start = cur_start.copy()     # must copy

        gradient = fderiv(cur_start)
        cur_start -= gradient * step_size   # move in opposite direction

        start_list.append(cur_start)
        iter += 1

    return cur_start


def trial1():
    def f(x, y):
        return 3 * (x + 2) ** 2 + (y - 1) ** 2       # 3(x + 2)² + (y - 1)²

    def fderiv_dx(x, y):
        return 6 * (x + 2)

    def fderiv_dy(x, y):
        return 2 * (y - 1)

    def fderiv(state):
        x, y = state[0], state[1]
        return np.array([fderiv_dx(x, y), fderiv_dy(x, y)])

    func_name = 'Gradient Descent on 2x² - 4x y + y⁴ + 2'

    inital_x, inital_y = -5.0, 2.5
    state = np.array([inital_x, inital_y])
    mn = gradient_descent(fderiv, state)
    print(f'The minimum found at state = {mn}')




if __name__ == '__main__':
    trial1()

