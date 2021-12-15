import numpy as np
import matplotlib.pyplot as plt
 
#x = np.linspace(0, 2 * np.pi, 10)
#y1, y2 = np.sin(x), np.cos(x)
 
x = np.array([0, 0.2, 0.4, 0.6, 0.8])

#plt.title('Subclass Accuracy')
#plt.title('Subclass MRR (filter) of different η')
#plt.ylabel('#Sub Acc')
#plt.ylabel('#Sub MRR (Raw)')

if 0:
	y1 = np.array([86.64, 85.12, 85.32, 78.55, 72.04])
	y2 = np.array([90.93, 88.63, 88.31, 83.53, 78.53])
	y3 = np.array([93.15, 88.33, 89.06, 83.96, 80.22])
	y4 = np.array([93.64, 90.05, 89.08, 85.99, 83.31])
	plt.ylabel('#Sub Accuracy', size=16)
else:
	y1 = np.array([3.42, 2.62, 2.23, 1.64, 1.74])
	y2 = np.array([6.40, 6.14, 7.08, 6.21, 4.95])
	y3 = np.array([10.13, 8.19, 7.70, 6.59, 6.61])
	y4 = np.array([10.14, 8.31, 7.72, 6.83, 7.02])
	plt.ylabel('#Sub MRR (Raw)', size=16)

plt.plot(x, y1, label = 'TransE', linestyle= 'dashed', marker='o')
plt.plot(x, y2, label = 'KEPLER', linestyle= 'dashed', marker='o')
plt.plot(x, y3, label = 'BERT-tiny', linestyle= 'dashed', marker='o')
plt.plot(x, y4, label = 'PSN_SELFATT', linestyle= 'dashed', marker='o')
plt.legend(prop={"size":10}, loc="upper left", bbox_to_anchor=(0.35,0.6))

plt.xlabel('Noise Rate', fontsize=20)
plt.xticks(x, fontsize=20)
plt.yticks(fontsize=20)
 
plt.show()

# adjust : left 25, bottom 45, right 90, top 88