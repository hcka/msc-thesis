# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 03:54:31 2018

@author: Admin
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math, os
fig, ax = plt.subplots(1)
mu = -3
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax.plot(x,mlab.normpdf(x, mu, sigma),'g',label='TN')
mu = 3
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax.plot(x,mlab.normpdf(x, mu, sigma),'r',label='TP')
ax.set_yticklabels([])
ax.set_xticklabels([])
#plt.plot([0.5, 0.5], [0, 0.4], 'k--')
plt.plot([0, 0], [0, 0.4], 'k', label='Threshold')
#plt.plot([-0.5, -0.5], [0, 0.4], 'k--',label='Limits')
plt.xlabel('Prediction result')
plt.ylabel('Number of images')
plt.title('Ideal threshold')
plt.legend(loc='upper right')
fig=plt.gcf()
plt.tight_layout(); plt.show()
fig.savefig(os.path.join('C:/Users/Admin/Desktop/cleaner', 'ideal_thresh.png'), dpi=100)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot([0, 0], [0, 1], 'b')
plt.plot([0, 1], [1, 1], 'b', label='AUC = 1')
plt.xlabel('False positive rate (1 - Specificitiy)')
plt.ylabel('True positive rate (Sensitivity)')
plt.title('Ideal ROC curve')
plt.legend(loc='best')
plt.grid()
fig=plt.gcf()
plt.tight_layout(); plt.show()
fig.savefig(os.path.join('C:/Users/Admin/Desktop/cleaner', 'ideal_roc.png'), dpi=100)