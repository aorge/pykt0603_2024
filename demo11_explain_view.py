import numpy as np

a = np.array([[1, 2], [3, 4]])
print(a)
b = a.view()
c = a
print(a.shape, b.shape, c.shape)
b.shape = (4, -1)
print(a.shape, b.shape, c.shape)
print(a)
print(b)
print(c)
c.shape = (-1, 4)
print(a.shape, b.shape, c.shape)
print(a)
print(b)
print(c)
print("change a[0][0] to 100")
a[0][0] = 100
print(a)
print(b)
print(c)