def gradient_ascent_with_momentum(starting_point_x, starting_point_y, initial_learning_rate, num_iterations, beta=0.9):
    x, y = starting_point_x, starting_point_y
    learning_rate = initial_learning_rate
    v_x, v_y = 0, 0  # Initialize velocity
    
    for iteration in range(num_iterations):
        grad_x, grad_y = gradient_function_2d(x, y)
        # Update velocity
        v_x = beta * v_x + (1 - beta) * grad_x
        v_y = beta * v_y + (1 - beta) * grad_y
        
        # Update parameters
        x += learning_rate * v_x
        y += learning_rate * v_y
        
    return x, y

max_x, max_y = gradient_ascent_with_momentum(starting_point_x, starting_point_y, initial_learning_rate, num_iterations)
max_value_2d = function_to_maximize_2d(max_x, max_y)

print(f'Maximum occurs at (x, y) = ({max_x}, {max_y}), with a value of f(x, y) = {max_value_2d}.')
