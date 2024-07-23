import itertools

def conway(xs):
    # a b c
    # d e f
    # g h i
    a,b,c,d,e,f,g,h,i = xs

    # neighbours ∈ [0,8]
    neighbours = a + b + c + d + f + g + h + i
    # alive ∈ [0,1]
    alive = e

    if neighbours == 3 or alive and neighbours == 2:
        return 1
    else:
        return 0

def relu(x):
    return x if x > 0 else 0

def relu_conway(xs):
    a,b,c,d,e,f,g,h,i = xs

    neighbours = a + b + c + d + f + g + h + i
    alive = e

    def equals(target, value):
        return 1 - relu(target - value) - relu(value - target)

    has_three = equals(3, neighbours)
    has_two = equals(2, neighbours)
    supported = relu(alive - 1 + has_two)

    return has_three + supported

def network_conway(xs):
    a,b,c,d,e,f,g,h,i = xs

    # a nice variable to use for first layer
    neighbours = a + b + c + d + f + g + h + i

    # Hidden layer 1
    p1 = relu(neighbours - 3)
    p2 = relu(3 - neighbours)
    q1 = relu(neighbours - 2)
    q2 = relu(2 - neighbours)
    alive = relu(e)

    # Hidden layer 2
    has_three = relu(1 - p1 - p2)
    has_two_and_alive = relu(alive - q1 - q2)

    # Output
    out = relu(has_three + has_two_and_alive)

    return out

if __name__ == "__main__":
    for xs in itertools.product((0,1), repeat=9):
        a = conway(xs)
        b = network_conway(xs)
        b = 1 if b > 0 else 0
        if a != b:
            print(f'conway({xs})={a}')
            print(f'nn_conway({xs})={b}')