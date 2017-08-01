from matplotlib import pyplot as plt
x = [1,2,4,8,16]
y_cpu = [166.55,307.40,592.89,None,None]
y_gpu = []
assert(len(y_gpu) == len(y_cpu))
l1 = plt.plot(x, y_cpu, "-g",label="CPU Time")
l2 = plt.plot(x, y_gpu, "-r",label="GPU Time")
plt.xlabel("Epoch")
plt.ylabel("Time in sec")
plt.title("GPU vs CPU Training Time")
plt.legend(loc="upper center", shadow=True)
plt.show()