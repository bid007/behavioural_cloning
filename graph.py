from matplotlib import pyplot as plt
x = [1,2,4,8,16]
y_cpu = [166.55,307.40,592.89,1390.14,2952.36]
y_gpu = [26.944,48.314,95.13,190.612,378.51]
y_validation_loss = map(lambda x: 100*x, [0.0698,0.0650,0.0568,0.0702,0.0644,0.0537,0.0626,0.0586,\
0.0627,0.0621,0.0600,0.0570,0.0559,0.0579,0.0569,0.0577,0.0591,0.0542,0.0623,0.0719])
y_loss = map(lambda x: 100*x,[0.0306,0.0082,0.0069,0.0063,0.0061,0.0055,0.0053,0.0050,0.0051,\
0.0051,0.0048,0.0047,0.0045,0.0047,0.0047,0.0045,0.0044,0.0045,0.0044,0.0042])
assert(len(y_gpu) == len(y_cpu))

l1 = plt.plot(x, y_cpu, "-g",label="CPU Time")
l2 = plt.plot(x, y_gpu, "-r",label="GPU Time")
plt.xlabel("Epoch")
plt.ylabel("Time in sec")
plt.title("GPU vs CPU Training Time")
plt.legend(loc="upper center", shadow=True)
plt.show()

l1 = plt.plot(range(1,21), y_validation_loss, "-g",label="Validation Loss")
l2 = plt.plot(range(1, 21), y_loss, "-r",label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend(loc="upper center", shadow=True)
plt.show()