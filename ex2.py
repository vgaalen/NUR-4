import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import G

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
potential = 4*np.pi*G*densitycontrast
fig, ax = plt.subplots(2,2, figsize=(10,8))
pcm = ax[0,0].pcolormesh(np.arange(0,16), np.arange(0,16), potential[4])
ax[0,0].set(ylabel='y', title='z = 4.5')
fig.colorbar(pcm, ax=ax[0,0], label='Potential')
pcm =ax[0,1].pcolormesh(np.arange(0,16), np.arange(0,16), potential[9])
ax[0,1].set(title='z = 9.5')
fig.colorbar(pcm, ax=ax[0,1], label='Potential')
pcm = ax[1,0].pcolormesh(np.arange(0,16), np.arange(0,16), potential[11])
ax[1,0].set(ylabel='y', xlabel='x', title='z = 11.5')
fig.colorbar(pcm, ax=ax[1,0], label='Potential')
pcm = ax[1,1].pcolormesh(np.arange(0,16), np.arange(0,16), potential[14])
ax[1,1].set(xlabel='x', title='z = 14.5')
fig.colorbar(pcm, ax=ax[1,1], label='Potential')
ax[0,0].set_aspect('equal', 'box')
ax[0,1].set_aspect('equal', 'box')
ax[1,0].set_aspect('equal', 'box')
ax[1,1].set_aspect('equal', 'box')
plt.savefig("fig2b_pot.png")
plt.close()

def bit_reverse(n, decimal_places = 10):
    if n%1==0:
        #print('int')
        # if n is an integer
        n = int(n)
        #print(n, bin(n), bin(n)[:1:-1])

        return int(bin(n)[:1:-1], 2)
    
    # if n is a float

    # split the number at the decimal point
    whole, dec = np.format_float_positional(n, precision=decimal_places).split(".")

    whole, dec = int(whole), int(dec)
    print(bin(whole), bin(dec))

    #res = bin(whole).lstrip("0b") + "."

    whole = int(bin(whole)[:-2:-1], 2)
    dec = int(bin(dec)[:-2:-1], 2)
    return float(str(whole)+"."+str(dec))

def test_bit_reverse():
    # x = np.random.randint(1,1e5)
    # try:
    #     assert(np.invert(x) == bit_reverse(x))
    # except:
    #     print(x, np.invert(x), bit_reverse(x))
    #     assert(np.invert(x) == bit_reverse(x))
    
    try:
        assert(bit_reverse(int('10011101', 2) == int('10111001', 2)))
    except:
        print('abc', int('10011101', 2), int('10111001', 2), bit_reverse(int('10011101', 2)))
        assert(bit_reverse(int('10011101', 2) == int('10111001', 2)))


def fft(x):
    """
    Perform a 3D Fast Fourier Transform
    """

    # TODO: pad the array so the length of all axes is a power of 2

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j,:] = _fft(x[i,j])
    
    for i in range(x.shape[0]):
        for k in range(x.shape[2]):
            x[i,:,k] = _fft(x[i,:,k])
    
    for j in range(x.shape[1]):
        for k in range(x.shape[2]):
            x[:,j,k] = _fft(x[:,j,k])
    
    return x

def _fft(x):
    """
    Perform a 1D Fast Fourier Transform
    """
    
    N = len(x)

    # bit-reversal
    ind = []
    for i in range(N):
        ind.append(bit_reverse(i))

    y = np.zeros(x.shape)
    y[ind] = x
    
    for Nj in range(2,N,2):
        for n in range(0,N-1,Nj):
            for k in range(0,Nj//2-1):
                print(f"Nj: {Nj}, n: {n}, k: {k}, m: {n+k}")
                m = n+k
                if m+Nj//2 >= N:
                    continue

                t = y[m]
                y[m] = t + np.exp(2j*np.pi*k/Nj) * y[m+Nj//2]
                y[m+Nj//2] = t - np.exp(2j*np.pi*k/Nj) * y[m+Nj//2]
    return y

def _dft(x):
    N = len(x)

    if N>2:
        x = np.concatenate((_dft(x[::2],N=N/2),_dft(x[1::2],N=N/2)))
    
    for k in range(N/2-1):
        t = x[k]
        x[k] = t + np.exp(2j*np.pi*k/N) * x[k+N/2]
        x[k+N/2] = t - np.exp(2j*np.pi*k/N) * x[k+N/2]
    return x

#test_bit_reverse()

fourier_potential = np.abs(np.fft.fftn(potential))
fourier_potential_itt = np.abs(fft(potential.value))
#fourier_potential_non_itt = np.abs(_dft(potential))
fig, ax = plt.subplots(2,2, figsize=(10,8))
pcm = ax[0,0].pcolormesh(np.arange(0,16), np.arange(0,16), fourier_potential[4])
ax[0,0].set(ylabel='y', title='z = 4.5')
fig.colorbar(pcm, ax=ax[0,0], label=r'log10(|$\~\Phi$|)')
pcm =ax[0,1].pcolormesh(np.arange(0,16), np.arange(0,16), fourier_potential[9])
ax[0,1].set(title='z = 9.5')
fig.colorbar(pcm, ax=ax[0,1], label=r'log10(|$\~\Phi$|)')
pcm = ax[1,0].pcolormesh(np.arange(0,16), np.arange(0,16), fourier_potential[11])
ax[1,0].set(ylabel='y', xlabel='x', title='z = 11.5')
fig.colorbar(pcm, ax=ax[1,0], label=r'log10(|$\~\Phi$|)')
pcm = ax[1,1].pcolormesh(np.arange(0,16), np.arange(0,16), fourier_potential[14])
ax[1,1].set(xlabel='x', title='z = 14.5')
fig.colorbar(pcm, ax=ax[1,1], label=r'log10(|$\~\Phi$|)')
ax[0,0].set_aspect('equal', 'box')
ax[0,1].set_aspect('equal', 'box')
ax[1,0].set_aspect('equal', 'box')
ax[1,1].set_aspect('equal', 'box')
plt.savefig("fig2b_fourier.png")
plt.close()

fig, ax = plt.subplots(2,2, figsize=(10,8))
pcm = ax[0,0].pcolormesh(np.arange(0,16), np.arange(0,16), fourier_potential[4])
ax[0,0].set(ylabel='y', title='z = 4.5')
fig.colorbar(pcm, ax=ax[0,0], label=r'log10(|$\~\Phi$|)')
pcm =ax[0,1].pcolormesh(np.arange(0,16), np.arange(0,16), fourier_potential_itt[4])
ax[0,1].set(title='z = 9.5')
fig.colorbar(pcm, ax=ax[0,1], label=r'log10(|$\~\Phi$|)')
pcm = ax[1,0].pcolormesh(np.arange(0,16), np.arange(0,16), fourier_potential[14])
ax[1,0].set(ylabel='y', xlabel='x', title='z = 11.5')
fig.colorbar(pcm, ax=ax[1,0], label=r'log10(|$\~\Phi$|)')
pcm = ax[1,1].pcolormesh(np.arange(0,16), np.arange(0,16), fourier_potential_itt[14])
ax[1,1].set(xlabel='x', title='z = 14.5')
fig.colorbar(pcm, ax=ax[1,1], label=r'log10(|$\~\Phi$|)')
ax[0,0].set_aspect('equal', 'box')
ax[0,1].set_aspect('equal', 'box')
ax[1,0].set_aspect('equal', 'box')
ax[1,1].set_aspect('equal', 'box')
plt.savefig("fig2b_fourier-1.png")
plt.close()
