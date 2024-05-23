from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
from astropy import units as u
from astropy.constants import G, M_sun
import numpy as np
import matplotlib.pyplot as plt

# Question 1: Simulating the solar system

# pick a time (please use either this or the current time)
t = Time("2024-05-23 23:59")

# Problem 1.a
x_init, y_init, z_init = np.zeros((3,9))
vx_init, vy_init, vz_init = np.zeros((3,9))
names = np.array(['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'])
for i, name in enumerate(names):
    initial_conditions = get_body_barycentric_posvel(name, t)
    x_init[i], y_init[i], z_init[i] = initial_conditions[0].xyz.to_value(u.AU)
    vx_init[i], vy_init[i], vz_init[i] = initial_conditions[1].xyz.to_value(u.AU/u.d)

fig, ax = plt.subplots(1,2, figsize=(12,5), constrained_layout=True)
for i, obj in enumerate(names):
    ax[0].scatter(x_init[i], y_init[i], label=obj)
    ax[1].scatter(x_init[i], z_init[i], label=obj)
ax[0].set_aspect('equal', 'box')
ax[1].set_aspect('equal', 'box')
ax[0].set(xlabel='X [AU]', ylabel='Y [AU]')
ax[1].set(xlabel='X [AU]', ylabel='Z [AU]')
plt.legend(loc=(1.05,0))
plt.savefig("fig1a.png")
plt.close()

# Problem 1.b
# For visibility, you may want to do two versions of this plot: 
# one with all planets, and another zoomed in on the four inner planets

def a(x):
    r2 = x[0]**2+x[1]**2+x[2]**2
    abs = G*M_sun/r2
    unit = x/np.sqrt(r2)
    return unit[0]*abs, unit[1]*abs, unit[2]*abs

time = t + np.arange(0,200*365,0.5)*u.day
h = 0.5 * u.day

x, y, z = np.zeros((3,9,len(time))) * u.AU
x[:,0], y[:,0], z[:,0] = x_init.copy() * u.AU, y_init.copy() *u.AU, z_init.copy()*u.AU
vx, vy, vz = np.zeros((3,9,len(time))) * u.AU/u.d
vx[:,0], vy[:,0], vz[:,0] = vx_init.copy() * u.AU/u.d, vy_init.copy() * u.AU/u.d, vz_init.copy() * u.AU/u.d

for i in range(1,len(names)):
    print(names[i])
    for j in range(0,len(time)-1):
        ax, ay, az = a(np.array([(x[0,j]-x[i,j]).to_value(u.AU),(y[0,j]-y[i,j]).to_value(u.AU),(z[0,j]-z[i,j]).to_value(u.AU)]))

        if j==0:
            vx[i,j+1] = vx[i,j] + h/2 * ax / u.AU**2
            vy[i,j+1] = vy[i,j] + h/2 * ay / u.AU**2
            vz[i,j+1] = vz[i,j] + h/2 * az / u.AU**2
        else:
            vx[i,j+1] = vx[i,j] + h * ax / u.AU**2
            vy[i,j+1] = vy[i,j] + h * ay / u.AU**2
            vz[i,j+1] = vz[i,j] + h * az / u.AU**2

        x[i,j+1] = x[i,j] + h * vx[i,j+1]
        y[i,j+1] = y[i,j] + h * vy[i,j+1]
        z[i,j+1] = z[i,j] + h * vz[i,j+1]

fig, ax = plt.subplots(1,2, figsize=(12,5), constrained_layout=True)
for i, obj in enumerate(names):
    ax[0].plot(x[i,:], y[i,:], label=obj)
    ax[1].plot(time.to_value('jyear'), z[i,:], label=obj)
ax[0].set_aspect('equal', 'box')
ax[0].set(xlabel='X [AU]', ylabel='Y [AU]')
ax[1].set(xlabel='Time [yr]', ylabel='Z [AU]')
plt.legend(loc=(1.05,0))
plt.savefig("fig1b.png")
plt.close()

fig, ax = plt.subplots(1,2, figsize=(12,5), constrained_layout=True)
for i, obj in enumerate(names[:5]):
    ax[0].plot(x[i,:], y[i,:], label=obj)
    ax[1].plot(time.to_value('jyear'), z[i,:], label=obj)
ax[0].set_aspect('equal', 'box')
ax[0].set(xlabel='X [AU]', ylabel='Y [AU]')
ax[1].set(xlabel='Time [yr]', ylabel='Z [AU]')
plt.legend(loc=(1.05,0))
plt.savefig("fig1b-1.png")
plt.close()

fig, ax = plt.subplots(1,2, figsize=(12,5), constrained_layout=True)
for i, obj in enumerate(names[:5]):
    ax[0].plot(vx[i,:], vy[i,:], label=obj)
    ax[1].plot(time.to_value('jyear'), vz[i,:], label=obj)
ax[0].set_aspect('equal', 'box')
ax[0].set(xlabel='VX [AU]', ylabel='VY [AU]')
ax[1].set(xlabel='Time [yr]', ylabel='VZ [AU]')
plt.legend(loc=(1.05,0))
plt.savefig("fig1b-2.png")
plt.close()