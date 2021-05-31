from matplotlib import pyplot as plt
import numpy as np
import pickle

d_loss_list = open('d_loss_list.pkl', 'rb')
g_loss_list = open('g_loss_list.pkl', 'rb')

d_loss = pickle.load(d_loss_list)
g_loss = pickle.load(g_loss_list)

d_array = np.array(d_loss)
d_array = d_array.reshape(-1)
g_array = np.array(g_loss)
g_array = g_array.reshape(-1)

print(d_array)
print(g_array)
x = np.arange(0, len(d_array) * 50, 50)
# plt.plot(x, d_array)
plt.plot(x, g_array)
plt.xlabel('batch')
# plt.ylabel('discriminator loss')
plt.ylabel('generator loss')
plt.title('Loss curve')

plt.show()
