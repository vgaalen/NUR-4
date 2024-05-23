import numpy as np
import matplotlib.pyplot as plt

# Question 2: Calculating forces with the FFT

np.random.seed(121) # DO NOT CHANGE (so positions are the same for all students)

n_mesh = 16
n_part = 1024
positions = np.random.uniform(low=0, high=n_mesh, size=(3, n_part))

grid = np.arange(n_mesh) + 0.5
densities = np.zeros(shape=(n_mesh, n_mesh, n_mesh))
cellvol = 1.

for p in range(n_part):
    cellind = np.zeros(shape=(3, 2))
    dist = np.zeros(shape=(3, 2))

    for i in range(3):
        cellind[i] = np.where((abs(positions[i, p] - grid) < 1) |
                              (abs(positions[i, p] - grid - 16) < 1) | 
                              (abs(positions[i, p] - grid + 16) < 1))[0]
        dist[i] = abs(positions[i, p] - grid[cellind[i].astype(int)])

    cellind = cellind.astype(int)

    for (x, dx) in zip(cellind[0], dist[0]):    
        for (y, dy) in zip(cellind[1], dist[1]):
            for (z, dz) in zip(cellind[2], dist[2]):
                if dx > 15: dx = abs(dx - 16)
                if dy > 15: dy = abs(dy - 16)
                if dz > 15: dz = abs(dz - 16)

                densities[x, y, z] += (1 - dx)*(1 - dy)*(1 - dz) / cellvol

# Problem 2.a
mean_density = np.mean(densities)
densitycontrast = (densities - mean_density)/mean_density
fig, ax = plt.subplots(2,2, figsize=(10,8))
pcm = ax[0,0].pcolormesh(np.arange(0,16), np.arange(0,16), densitycontrast[4])
ax[0,0].set(ylabel='y', title='z = 4.5')
fig.colorbar(pcm, ax=ax[0,0], label='Density')
pcm =ax[0,1].pcolormesh(np.arange(0,16), np.arange(0,16), densitycontrast[9])
ax[0,1].set(title='z = 9.5')
fig.colorbar(pcm, ax=ax[0,1], label='Density')
pcm = ax[1,0].pcolormesh(np.arange(0,16), np.arange(0,16), densitycontrast[11])
ax[1,0].set(ylabel='y', xlabel='x', title='z = 11.5')
fig.colorbar(pcm, ax=ax[1,0], label='Density')
pcm = ax[1,1].pcolormesh(np.arange(0,16), np.arange(0,16), densitycontrast[14])
ax[1,1].set(xlabel='x', title='z = 14.5')
fig.colorbar(pcm, ax=ax[1,1], label='Density')
ax[0,0].set_aspect('equal', 'box')
ax[0,1].set_aspect('equal', 'box')
ax[1,0].set_aspect('equal', 'box')
ax[1,1].set_aspect('equal', 'box')
plt.savefig("fig2a.png")
plt.close()

# Problem 2.b
def bit_reverse_int(n, bits=None):
    """
    Perform a bit-reversal on an integer
    """
    if n == 0:
        return 0

    if bits is None:
        bits = int(np.log2(n))+1

    binary = bin(n)[2:]
    if len(binary) == bits:
        return int(binary[::-1], 2)
    else:
        add = bits - len(binary)
        if add<0:
            raise ValueError("Number of bits is too small")
        binary = binary.zfill(bits)
        return int(binary[::-1], 2)

def fft(x):
    """
    Perform a Fast Fourier Transform
    """
    x = x.astype(np.complex128)
    
    if len(x.shape) == 1:
        if x.shape[0] != 2**int(np.log2(x.shape[0])):
            x = np.pad(x, [0, 2**int(np.ceil(np.log2(x.shape[0]))) - x.shape[0]], mode='constant')
        return _fft(x)
    
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j,:] = _fft(x[i,j,:])
    
    for i in range(x.shape[0]):
        for k in range(x.shape[2]):
            x[i,:,k] = _fft(x[i,:,k])
    
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            x[:,j,k] = _fft(x[:,j,k])
    
    return x

def ifft(x):
    """
    Perform an Inverse Fast Fourier Transform
    """
    x = x.astype(np.complex128)
    
    if len(x.shape) == 1:
        if x.shape[0] != 2**int(np.log2(x.shape[0])):
            x = np.pad(x, [0, 2**int(np.ceil(np.log2(x.shape[0]))) - x.shape[0]], mode='constant')
        return _fft(x)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j,:] = _ifft(x[i,j,:])
    
    for i in range(x.shape[0]):
        for k in range(x.shape[2]):
            x[i,:,k] = _ifft(x[i,:,k])
    
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            x[:,j,k] = _ifft(x[:,j,k])
    
    return x

def _fft(x):
    """
    Perform a 1D Fast Fourier Transform
    """   
    N = len(x)

    # bit-reversal of indices
    bits = int(np.log2(N))
    ind = []
    for i in range(N):
        ind.append(bit_reverse_int(i, bits=bits))
    y = np.zeros(x.shape, dtype=np.complex128)
    y[ind] = x

    for Nj in 2**np.arange(1,int(bits+1)):
        Nj = int(Nj)
        for n in range(0,N,Nj):
            for k in range(0,Nj//2):
                m = n+k
                t = y[m]
                u = np.exp(-2j*np.pi*k/Nj) * y[m+Nj//2]
                y[m] = t + u
                y[m+Nj//2] = t - u
    return y

def _ifft(x):
    """
    Perform a 1D Fast Fourier Transform
    """    
    N = len(x)

    # bit-reversal of indices
    bits = int(np.log2(N))
    ind = []
    for i in range(N):
        ind.append(bit_reverse_int(i, bits=bits))
    y = np.zeros(x.shape, dtype=np.complex128)
    y[ind] = x

    for Nj in 2**np.arange(1,int(bits+1)):
        Nj = int(Nj)
        for n in range(0,N,Nj):
            for k in range(0,Nj//2):
                m = n+k
                t = y[m]
                u = np.exp(2j*np.pi*k/Nj) * y[m+Nj//2]
                y[m] = t + u
                y[m+Nj//2] = t - u
    return y/N

k = np.arange(densitycontrast.shape[0])
kx, ky, kz = np.meshgrid(k, k, k)
k2 = kx**2 + ky**2 + kz**2
k2[0,0,0] = 1
k2 = np.divide(k2, np.max(k2), casting='safe')

fourier_potential = fft(densitycontrast)
gravitational_field = 4*np.pi*mean_density * ifft(fourier_potential/k2)

fig, ax = plt.subplots(2,2, figsize=(10,8))
pcm = ax[0,0].pcolormesh(np.arange(0,16), np.arange(0,16), np.log10(np.abs(fourier_potential[4])))
ax[0,0].set(ylabel='y', title='z = 4.5')
fig.colorbar(pcm, ax=ax[0,0], label=r'log10(|$\~\Phi$|)')
pcm =ax[0,1].pcolormesh(np.arange(0,16), np.arange(0,16), np.log10(np.abs(fourier_potential[9])))
ax[0,1].set(title='z = 9.5')
fig.colorbar(pcm, ax=ax[0,1], label=r'log10(|$\~\Phi$|)')
pcm = ax[1,0].pcolormesh(np.arange(0,16), np.arange(0,16), np.log10(np.abs(fourier_potential[11])))
ax[1,0].set(ylabel='y', xlabel='x', title='z = 11.5')
fig.colorbar(pcm, ax=ax[1,0], label=r'log10(|$\~\Phi$|)')
pcm = ax[1,1].pcolormesh(np.arange(0,16), np.arange(0,16), np.log10(np.abs(fourier_potential[14])))
ax[1,1].set(xlabel='x', title='z = 14.5')
fig.colorbar(pcm, ax=ax[1,1], label=r'log10(|$\~\Phi$|)')
ax[0,0].set_aspect('equal', 'box')
ax[0,1].set_aspect('equal', 'box')
ax[1,0].set_aspect('equal', 'box')
ax[1,1].set_aspect('equal', 'box')
plt.savefig("fig2b_fourier.png")
plt.close()

fig, ax = plt.subplots(2,2, figsize=(10,8))
pcm = ax[0,0].pcolormesh(np.arange(0,16), np.arange(0,16), np.log10(np.abs(gravitational_field[4])))
ax[0,0].set(ylabel='y', title='z = 4.5')
fig.colorbar(pcm, ax=ax[0,0], label=r'log10(|$\Phi$|)')
pcm =ax[0,1].pcolormesh(np.arange(0,16), np.arange(0,16), np.log10(np.abs(gravitational_field[9])))
ax[0,1].set(title='z = 9.5')
fig.colorbar(pcm, ax=ax[0,1], label=r'log10(|$\Phi$|)')
pcm = ax[1,0].pcolormesh(np.arange(0,16), np.arange(0,16), np.log10(np.abs(gravitational_field[11])))
ax[1,0].set(ylabel='y', xlabel='x', title='z = 11.5')
fig.colorbar(pcm, ax=ax[1,0], label=r'log10(|$\Phi$|)')
pcm = ax[1,1].pcolormesh(np.arange(0,16), np.arange(0,16), np.log10(np.abs(gravitational_field[14])))
ax[1,1].set(xlabel='x', title='z = 14.5')
fig.colorbar(pcm, ax=ax[1,1], label=r'log10(|$\Phi$|)')
ax[0,0].set_aspect('equal', 'box')
ax[0,1].set_aspect('equal', 'box')
ax[1,0].set_aspect('equal', 'box')
ax[1,1].set_aspect('equal', 'box')
plt.savefig("fig2b_grav.png")
plt.close()