
from matplotlib import pyplot as plt

a= 5
plt.plot(range(10), range(10), label="test label")
a = a*2
print(a)
plt.plot(range(10), [5 for x in range(10)], label="another test")

plt.legend().set_draggable(True)

plt.show()