# y 1000, 10000, 10000000
y = 10000000


def calculate(x):
    for i in range(0, 1000000):
        x += 0.0000001
    x -= 0.1
    return x


print("%.6f" % calculate(y))
