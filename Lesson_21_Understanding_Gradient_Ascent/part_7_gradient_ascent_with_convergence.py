def gradient_ascent_with_stop_criteria(starting_point_x, starting_point_y, initial_learning_rate, num_iterations, tolerance=1e-6):
    x, y = starting_point_x, starting_point_y
    learning_rate = initial_learning_rate
    
    for iteration in range(num_iterations):
        grad_x, grad_y = gradient_function_2d(x, y)
        
        # Check for convergence
        if np.sqrt(grad_x**2 + grad_y**2) < tolerance:
            print(f'Stopping early at iteration {iteration}, gradient too small.')
            break
        
        x += learning_rate * grad_x
        y += learning_rate * grad_y
        
    return x, y

max_x, max_y = gradient_ascent_with_stop_criteria(starting_point_x, starting_point_y, initial_learning_rate, num_iterations)
max_value_2d = function_to_maximize_2d(max_x, max_y)

print(f'Maximum occurs at (x, y) = ({max_x}, {max_y}), with a value of f(x, y) = {max_value_2d}.')
