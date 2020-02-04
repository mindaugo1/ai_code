x = 2
y = 5
learning_rate = 0.2

w = 0.2
b = 2
print(f'initial m: {w}')
guess = x*w+b
print(f'initial guess: {guess}')
error = y - guess
print(f'initial error: {error}')
error_to_compare = error


w = w + error * learning_rate
guess = x * w + b
error = y - guess

print(f'new_w: {w}')
print(f'updated guess: {guess}')
print(f'updated error: {error}')
