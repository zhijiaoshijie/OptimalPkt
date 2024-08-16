from itertools import islice

# Define a simple generator
def my_generator():
    yield 1
    yield 2
    yield 3
    yield 4
    yield 5

# Create a generator object
for i, x in enumerate(my_generator()):
    print(x)
    if i>2: break
for i, x in enumerate(my_generator()):
    print(x)
    if i>2: break
