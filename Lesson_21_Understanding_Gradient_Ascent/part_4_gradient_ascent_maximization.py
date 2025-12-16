def function_to_maximize_2d(x, y):
    return -((x - 2)**2 + (y - 3)**2) + 10

def gradient_function_2d(x, y):
    dfdx = -2 * (x - 2)
    dfdy = -2 * (y - 3)
    return dfdx, dfdy

def gradient_ascent_2d(starting_point_x, starting_point_y, learning_rate, num_iterations):
    x, y = starting_point_x, starting_point_y
    for _ in range(num_iterations):
        grad_x, grad_y = gradient_function_2d(x, y)
        x += learning_rate * grad_x
        y += learning_rate * grad_y
    return x, y

# Parameters
starting_point_x = 0
starting_point_y = 0
learning_rate = 0.1
num_iterations = 30

max_x, max_y = gradient_ascent_2d(starting_point_x, starting_point_y, learning_rate, num_iterations)
max_value_2d = function_to_maximize_2d(max_x, max_y)

print(f'Maximum occurs at (x, y) = ({max_x}, {max_y}), with a value of f(x, y) = {max_value_2d}.')
