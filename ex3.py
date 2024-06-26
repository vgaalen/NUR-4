import numpy as np
import matplotlib.pyplot as plt
import itertools
import time

# Question 3: Spiral and elliptical galaxies

# Problem 3.a
data = np.loadtxt('data/galaxy_data.txt')
features = data[:,:4]

# Scale the features (so they have mean 0 and sdev 1)
features = (features - np.mean(features, axis=0)[None,:]) / np.std(features, axis=0)[None,:]

np.savetxt("3a.txt", features)

bin_class = np.linspace(-2,2,50)
fig, ax = plt.subplots(2,2, figsize=(10,8))
ax[0,0].hist(features[:,0], bins=bin_class)
ax[0,0].set(ylabel='N', xlabel=r'$\kappa_{CO}$')
ax[0,1].hist(features[:,1], bins=bin_class)
ax[0,1].set(xlabel='Color')
ax[1,0].hist(features[:,2], bins=bin_class)
ax[1,0].set(ylabel='N', xlabel='Extended')
ax[1,1].hist(features[:,3], bins=bin_class)
ax[1,1].set(xlabel='Emission line flux')
plt.savefig("fig3a.png")
plt.close()

truth = data[:,4]
features2 = features[truth==1]
fig, ax = plt.subplots(2,2, figsize=(10,8))
ax[0,0].hist(features2[:,0], bins=bin_class)
ax[0,0].set(ylabel='N', xlabel=r'$\kappa_{CO}$')
ax[0,1].hist(features2[:,1], bins=bin_class)
ax[0,1].set(xlabel='Color')
ax[1,0].hist(features2[:,2], bins=bin_class)
ax[1,0].set(ylabel='N', xlabel='Extended')
ax[1,1].hist(features2[:,3], bins=bin_class)
ax[1,1].set(xlabel='Emission line flux')
plt.savefig("fig3a-mask.png")
plt.close()

# Problem 3.b
def logistic_function(x, theta):
    return 1/(1 + np.exp(-np.dot(x, theta)))

def eval(x, theta):
    val = logistic_function(x, theta)
    out = np.zeros_like(val)
    out[val>0.5] = 1
    return out

def cost_function(x, y, theta):
    y_model = logistic_function(x,theta)
    mask1 = (y==1)&(y_model!=0)
    mask2 = (y==0)&(y_model!=1)
    if np.sum(mask1) > 0 and np.sum(mask2) > 0:
        cost = 0
        cost += -np.sum(y[mask1]*np.log(y_model[mask1]))
        cost += -np.sum((1-y[mask2])*np.log(1-y_model[mask2]))
        return cost
    else:
        print("ERROR")
        return np.inf

def multid_quick_sort(y,x):
    """
    Quick sort algorithm

    Parameters
    ----------
    y : list
        The values to sort with
    x : 2d numpy array
        The list that should be sorted based on the values in y
    """

    if x.shape[0] <= 1:
        return y, x
    else:
        # sort the first, middle, and last values to make sure the pivot is the maximum or minimum
        if y[len(y)//2]<y[0]:
            if y[-1]<=y[len(y)//2]:
                y[0], y[-1] = y[-1], y[0]
                x[0,:], x[-1,:] = x[-1,:].copy(), x[0,:].copy()
            elif y[-1]<y[0]:
                y[0], y[len(y)//2], y[-1] = y[len(y)//2], y[-1], y[0]
                x[0,:], x[len(y)//2,:], x[-1,:] = x[len(y)//2,:].copy(), x[-1,:].copy(), x[0,:].copy()
            else:
                y[0], y[len(y)//2] = y[len(y)//2], y[0]
                x[0,:], x[len(y)//2,:] = x[len(y)//2,:].copy(), x[0,:].copy()
        else:
            if y[-1]<=y[0]:
                y[0], y[len(y)//2], y[-1] = y[-1], y[0], y[len(y)//2]
                x[0,:], x[len(y)//2,:], x[-1,:] = x[-1,:].copy(), x[0,:].copy(), x[len(y)//2,:].copy()
            elif y[-1]<y[len(y)//2]:
                y[len(y)//2], y[-1] = y[-1], y[len(y)//2]
                x[len(y)//2,:], x[-1,:] = x[-1,:], x[len(y)//2,:].copy()

        i=0
        j=len(y)-1
        pivot = len(y)//2
        # sort the arrays relative to the pivot
        while i<j:
            if y[i]>=y[pivot]:
                if y[j]<=y[pivot]:
                    y[i], y[j] = y[j], y[i]
                    x[i,:], x[j,:] = x[j,:], x[i,:].copy()
                    if i==pivot:
                        pivot=j
                    elif j==pivot:
                        pivot=i
                    i+=1
                    j-=1
                else:
                    j-=1
            else:
                i+=1
                if y[j]<=y[pivot]:
                    j-=1
        
        # the pivot is in the right location now, sort the rest of the array
        start_y, start_x = multid_quick_sort(y[:pivot],x[:pivot,:])
        end_y, end_x = multid_quick_sort(y[pivot+1:],x[pivot+1:,:])
        return np.concatenate((start_y, [y[pivot]], end_y), axis=0), np.concatenate((start_x, [x[pivot,:]], end_x), axis=0)

def DownhillSimplex(f,x,target_accuracy, num_itt=1000):
    """
    Implementation of the Downhill Simplex minimalisation algorithm.
    """
    if x.shape[0]!=x.shape[1]+1:
        raise ValueError("The input array x should have shape (n+1,n)")
    
    y = [0]*x.shape[0]
    for i in range(len(y)):
        y[i] = f(x[i])

    cost = np.zeros(num_itt)
    
    for itt in range(num_itt):
        cost[itt] = y[0]
        # 1. order the points and calculate the mean (excluding the worst point)
        y,x = multid_quick_sort(y,x)
        mean = np.mean(x[:-1],axis=0)

        # 2. check if the fractional range in f(x) (no in x!), this is |f(x_N)-f(x_0)|/[0.5*|f(x_N)+f(x_0)|], is within target accuracy and if so terminate
        if np.isnan(y[-1]) is False and (y[0] == y[1] or np.abs(y[-1] - y[0]) / (0.5*np.abs(y[-1] + y[0]))) < target_accuracy:
            return x[0]
        
        # 3. propose a new point by reflecting x_N:x_try=2mean-x_N
        x_try = 2*mean - x[-1]
        if f(x[0])<=f(x_try)<f(x[-1]):
            x[-1] = x_try
            y[-1] = f(x_try)
        elif f(x_try)<f(x[0]):
            x_exp = 2*x_try-mean
            if f(x_exp)<f(x_try):
                x[-1] = x_exp
                y[-1] = f(x_exp)
            else:
                x[-1] = x_try
                y[-1] = f(x_try)
        else:
            x_try = 0.5*(mean+x[-1])
            if f(x_try)<f(x[-1]):
                x[-1] = x_try
                y[-1] = f(x_try)
            else:
                x[1:] = 0.5*(x[0]+x[1:])
                for i in range(1,len(y)):
                    y[i] = f(x[i])
    return x[0], cost

# Minimize the cost function
theta = np.ones(4)
x_0 = np.concatenate([[theta],np.random.rand(4,4)])

out = np.zeros((6,2))
out[0], cost12 = DownhillSimplex(lambda x: cost_function(features[:,:2], truth, x), x_0[:3,:2], 1e-5)
out[1], cost13 = DownhillSimplex(lambda x: cost_function(features[:,:3:2], truth, x), x_0[:3,:3:2], 1e-5)
out[2], cost14 = DownhillSimplex(lambda x: cost_function(features[:,::3], truth, x), x_0[:3,::3], 1e-5)
out[3], cost23 = DownhillSimplex(lambda x: cost_function(features[:,1:3], truth, x), x_0[:3,1:3], 1e-5)
out[4], cost24 = DownhillSimplex(lambda x: cost_function(features[:,1::2], truth, x), x_0[:3,1::2], 1e-5)
out[5], cost34 = DownhillSimplex(lambda x: cost_function(features[:,2:4], truth, x), x_0[:3,2:4], 1e-5)

features_subset = np.array([features[:,:2],features[:,:3:2],features[:,::3],features[:,1:3],features[:,1::2],features[:,2:4]])

fig, ax  = plt.subplots(1,1, figsize=(10,5), constrained_layout=True)
ax.plot(np.arange(0,len(cost12)), cost12[:], label='Features 1+2')
ax.plot(np.arange(0,len(cost13)), cost13[:], label='Features 1+3')
ax.plot(np.arange(0,len(cost14)), cost14[:], label='Features 1+4')
ax.plot(np.arange(0,len(cost23)), cost23[:], label='Features 2+3')
ax.plot(np.arange(0,len(cost34)), cost34[:], label='Features 3+4')
ax.plot(np.arange(0,len(cost24)), cost24[:], label='Features 2+4')
ax.set(xlabel='Number of iterations', ylabel='Cost function')
ax.set_xscale('log')
plt.legend(loc=(1.05,0))
plt.savefig("fig3b.png")
plt.close()

# Problem 3.c
bin_class = truth
fig, ax = plt.subplots(3,2,figsize=(10,15))
names = [r'$\kappa_{CO}$', 'Color', 'Extended', 'Emission line flux']
plot_idx = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]
for i, comb in enumerate(itertools.combinations(np.arange(0,4), 2)):
    #print(comb)
    ax[plot_idx[i][0],plot_idx[i][1]].scatter(features[:,comb[0]], features[:,comb[1]], c=bin_class)
    ax[plot_idx[i][0],plot_idx[i][1]].set(xlabel=names[comb[0]], ylabel=names[comb[1]])

    ylims = ax[plot_idx[i][0],plot_idx[i][1]].get_ylim()
    xlims = ax[plot_idx[i][0],plot_idx[i][1]].get_xlim()

    y_0 = -1 * out[i,1] * -2 / out[i,0]
    y_1 = -1 * out[i,1] * 2 / out[i,0]
    assert logistic_function(np.array([y_0,-2]),np.array(out[i])) == 1/2
    assert logistic_function(np.array([y_1,2]),np.array(out[i])) == 1/2
    ax[plot_idx[i][0], plot_idx[i][1]].axline(xy1=(y_0,-2), xy2=(y_1,2), color="black", linestyle=(0, (5, 5)))

    ax[plot_idx[i][0], plot_idx[i][1]].set_xlim(xlims)
    ax[plot_idx[i][0], plot_idx[i][1]].set_ylim(ylims)

plt.savefig("fig3c.png")
plt.close()


# Calculate true/flase positives/negatives
labels = ["1+2","1+3","1+4","2+3","2+4","3+4"]
with open("3c.txt", "w") as f:
    f.write("Parameters, True Positive, False Positive, True Negative, False Negative, F1-Score\n")
    for i in range(6):
        true_positive = np.sum((eval(features_subset[i], out[i]) == 1) & (truth == 1))
        false_positive = np.sum((eval(features_subset[i], out[i]) == 1) & (truth == 0))
        true_negative = np.sum((eval(features_subset[i], out[i]) == 0) & (truth == 0))
        false_negative = np.sum((eval(features_subset[i], out[i]) == 0) & (truth == 1))

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1 = 2 * precision * recall / (precision + recall)

        f.write(f"{labels[i]}, {true_positive}, {false_positive}, {true_negative}, {false_negative}, {f1}\n")
