import numpy as np


def printAll():
    for x in (a, b, c, d):
        print("標記:",x)


a = np.array([[1, 2], [3, 4]])
b = a
c = a.view()
d = a.copy()

#printAll()
print("change a[0][0]=100")
a[0][0] = 100
printAll()
print("change a shape")
a.shape = (4,)
#printAll()
