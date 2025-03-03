import numpy as np
from numpy.linalg import norm

def gradient_descent(fderiv, inital_start, step_size = 0.001, precision = 0.00001, max_iter = 1000):
    """
    Implements the gradient descent optimization algorithm.
    
    Args:
        fderiv: Function that calculates the gradient (derivative) of the objective function
        inital_start: Initial point to start the optimization
        step_size: Learning rate that controls the size of each descent step
        precision: Convergence threshold for stopping the algorithm
        max_iter: Maximum number of iterations to prevent infinite loops
    
    Returns:
        The point that minimizes the objective function
    """
    cur_start = np.array(inital_start)  # Convert to numpy array if not already
    last_start = cur_start + 100 * precision    # Initialize with a different value to enter the loop
    start_list = [cur_start]  # Keep track of all points visited
    iter = 0  # Iteration counter
    
    # Continue until the change is smaller than precision or maximum iterations reached
    while norm(cur_start - last_start) > precision and iter < max_iter:
        print(cur_start)  # Print current position
        last_start = cur_start.copy()     # Store current position (must copy to avoid reference issues)
        gradient = fderiv(cur_start)  # Calculate gradient at current position
        cur_start -= gradient * step_size   # Update position by moving opposite to gradient direction
        start_list.append(cur_start)  # Store the new position
        iter += 1  # Increment iteration counter
    
    return cur_start  # Return the final position (approximation of minimum)

def trial1():
    """
    Test function for gradient descent algorithm using the function f(x,y) = 3(x+2)² + (y-1)²
    The minimum of this function should be at x=-2, y=1
    """
    def f(x, y):
        return 3 * (x + 2) ** 2 + (y - 1) ** 2       # 3(x + 2)² + (y - 1)²
    
    def fderiv_dx(x, y):
        return 6 * (x + 2)  # Partial derivative with respect to x
    
    def fderiv_dy(x, y):
        return 2 * (y - 1)  # Partial derivative with respect to y
    
    def fderiv(state):
        """Returns the gradient vector [df/dx, df/dy] at the given state"""
        x, y = state[0], state[1]
        return np.array([fderiv_dx(x, y), fderiv_dy(x, y)])
    
    func_name = 'Gradient Descent on 3(x+2)² + (y-1)²'  # Note: The comment in the original code doesn't match the function
    inital_x, inital_y = -5.0, 2.5  # Starting point
    state = np.array([inital_x, inital_y])
    
    # Run gradient descent algorithm
    mn = gradient_descent(fderiv, state)
    print(f'The minimum found at state = {mn}')

if __name__ == '__main__':
    trial1()
