import numpy as np
import matplotlib.pyplot as plt
 
#x = np.linspace(0, 2 * np.pi, 10)
#y1, y2 = np.sin(x), np.cos(x)
 
x = np.array([1, 2, 4, 8])
y1 = np.array([93.63, 93.59, 93.53, 94.02])
y2 = np.array([82.39, 85.30, 90.65, 91.52])

#y3 = np.array([16.03, 16.13, 15.59, 15.67])
#y4 = np.array([3.40, 5.37, 9.28, 11.37])

y3 = np.array([10.16, 10.30, 10.14, 10.20])
y4 = np.array([2.64, 4.09, 5.90, 7.46])

#plt.plot(x, y1, label = 'w MT', linestyle= 'dashed', marker='o')
#plt.plot(x, y2, label = 'w/o MT', linestyle= 'dashed', marker='o')
#plt.plot(x, y3, label = 'w MT', linestyle= 'dashed', marker='o')
#plt.plot(x, y4, label = 'w/o MT', linestyle= 'dashed', marker='o')
#plt.legend()
####plt.title('Subclass Accuracy of different η')
####plt.title('Subclass MRR (Raw) of different η')
#plt.ylabel('#Sub Accuracy', size=16)
#plt.ylabel('#Sub MRR (Raw)', size=16)
#plt.xlabel('value of η', size=20)

if 0:
	plt.plot(x, y1, label = 'w MT', linestyle= 'dashed', marker='o')
	plt.plot(x, y2, label = 'w/o MT', linestyle= 'dashed', marker='o')
	plt.legend()
	plt.ylabel('#Sub Accuracy', size=16)
	plt.xlabel('value of η', size=20)
else:
	plt.plot(x, y3, label = 'w MT', linestyle= 'dashed', marker='o')
	plt.plot(x, y4, label = 'w/o MT', linestyle= 'dashed', marker='o')
	plt.legend()
	plt.ylabel('#Sub MRR (Raw)', size=16)
	plt.xlabel('value of η', size=20)

plt.legend(prop={"size":14})
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)
 
plt.show()

# adjust : left 25, bottom 40, right 90, top 88