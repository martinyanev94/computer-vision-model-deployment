def gradient_ascent_2d_with_schedule(starting_point_x, starting_point_y, initial_learning_rate, num_iterations):
    x, y = starting_point_x, starting_point_y
    learning_rate = initial_learning_rate
    
    for iteration in range(num_iterations):
        grad_x, grad_y = gradient_function_2d(x, y)
        x += learning_rate * grad_x
        y += learning_rate * grad_y
        
        # Update the learning rate
        learning_rate = initial_learning_rate / (1 + iteration / 10)
    return x, y

# Parameters
initial_learning_rate = 0.1
num_iterations = 30

max_x, max_y = gradient_ascent_2d_with_schedule(starting_point_x, starting_point_y, initial_learning_rate, num_iterations)
max_value_2d = function_to_maximize_2d(max_x, max_y)

print(f'Maximum occurs at (x, y) = ({max_x}, {max_y}), with a value of f(x, y) = {max_value_2d}.')
