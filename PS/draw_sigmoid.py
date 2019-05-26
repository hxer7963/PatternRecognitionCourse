import numpy as np
from numpy import exp
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))
    
def partial_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def progressive_line(x):
    return 1.0

axis_x = np.arange(-8, 8, 0.01)
sigmoid_y = sigmoid(axis_x)
sigmoid_partial_y = partial_sigmoid(axis_x)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

plt.xlim(-6, 6)
plt.ylim(0, 1.1)
 
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
 
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.set_xticks([-5,-3,0,3,5])
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
ax.set_yticks([0.25, 0.5, 0.75, 1])

plt.plot(axis_x, sigmoid_y, color="b", label='$\sigma(x)$')
plt.plot(axis_x, sigmoid_partial_y, color='r', label="$\sigma^{'}(x)$")
# plt.plot(axis_x, progressive_line, color='k', linestyle=':')
print(partial_sigmoid(3), partial_sigmoid(5))

plt.legend(loc='upper left')

plt.title("$\sigma(x)$ vs. $\sigma^{'}(x)$")

plt.savefig('sigmoid.jpeg')
plt.show()