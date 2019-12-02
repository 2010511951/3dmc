import os
import sys
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import math

class point:
	def __init__(self, x, y, z ):
		if x==None and z==None:
			self.x = 0
			self.y = 0
			self.z = 0
		else:
			self.x = x
			self.y = y
			self.z = z

class k:
	def __init__(self, theta, fac):
		self.theta = theta
		self.fac = fac
		self.x = math.sin(theta)*math.cos(fac)
		self.y = math.sin(theta)*math.sin(fac)
		self.z = math.cos(theta)

# define G1 function for curve fitting
def g1_func(tau, ua, us_prime, S0, BFI):
	
	#ua = 0.002
	#us_prime = 1

	zb = 1.76/us_prime
	z0 = 1/us_prime
	rho = 20 # unit: mm
	r1 = np.sqrt(rho**2 + z0**2)
	r2 = np.sqrt(rho**2 + (z0+2*zb)**2 )
	k0 = 2*math.pi/(800*1e-6)  # unit: mm
	K = np.sqrt(3*ua*us_prime + (us_prime* k0)**2 * (6*BFI*tau) )
	
	
	return S0*3*us_prime/4/math.pi*(np.exp(-K*r1)/r1 - np.exp(-K*r2)/r2)

simul_set = sorted(glob(os.path.join('output/skull/thickness1mm/', '*.npy')))

simul_results = []
for file in simul_set:
	print(file)
	photon_set = np.load(file)
	simul_results.extend(photon_set)
print('simulation results loaded')
print('detected photon number: {}'.format(len(simul_results)))

tau=np.logspace(-14, -1, 2000)
G1=np.zeros([len(tau)],dtype=float)
G1_c=np.zeros([len(tau)],dtype=float)
G1_d=np.zeros([len(tau)],dtype=float)
vmax = 6 # unit: mm/s
R = 0.05 # unit: mm
alpha = 1e-6 # unit: mm^2
total_vessel_num = 28800

g_conv = np.zeros([len(simul_results),1])
g_diffu = np.zeros([len(simul_results),1])
g_absorption = np.zeros([len(simul_results),1])

for N in range(len(simul_results)):
	g_conv_for_each_vessel = np.zeros([total_vessel_num,1])
	for s in range(len(simul_results[N])):
		if simul_results[N][s]['vessel_num'] >= 1:
			v = vmax*(1-(simul_results[N][s]['r']/R)**2)
			#v = 0
			#print(simul_results[N][s]['r'])
			D = 2*alpha*vmax*simul_results[N][s]['r']/(R**2)
			#D = 0
			g_conv_for_each_vessel[ int(simul_results[N][s]['vessel_num']) -1 ] += simul_results[N][s]['q_y']*v
			#print(simul_results[N][s]['q_y'],simul_results[N][s]['q_m'])
			g_diffu[N] += simul_results[N][s]['q_m']**2*D
		g_absorption[N] += -simul_results[N][s]['ua']*simul_results[N][s]['L']
	g_conv[N] = np.sum(g_conv_for_each_vessel**2)

#print(g_conv)
#print(g_diffu)

# overall G1
for i, t in enumerate(tau):
	G1[i] = np.sum( np.exp(np.multiply(g_conv, -0.5*t**2)) * np.exp( np.multiply(g_diffu, -t) ) * np.exp(g_absorption) )
scale = np.sum(np.exp(g_absorption))
G1 = G1/scale

# diffusive G1
for i, t in enumerate(tau):
	G1_d[i] = np.sum(  np.exp( np.multiply(g_diffu, -t) ) * np.exp(g_absorption) )
G1_d = G1_d/scale

# convective G1
for i, t in enumerate(tau):
	G1_c[i] = np.sum( np.exp(np.multiply(g_conv, -0.5*t**2)) * np.exp(g_absorption) )
G1_c = G1_c/scale

plt.figure(1)
plt.plot(np.log10(tau),G1, label = 'overall')
plt.plot(np.log10(tau), G1_d, label = 'diffusive')
plt.plot(np.log10(tau), G1_c, label = 'convective')
plt.legend()
plt.show()

plt.figure(2)
tau = np.array(tau, dtype=np.float)
G1 = np.array(G1, dtype=np.float)
popt1, pconv = curve_fit(g1_func, tau, G1, [0.002, 1, 6000, 1000], maxfev = 20000000)
print('popt1_oerall', popt1)
G1_d = np.array(G1_d, dtype=np.float)
popt2, pconv = curve_fit(g1_func, tau, G1_d, [0.002, 1, 6000, 1000], maxfev = 20000000)
print('popt1_diffusive', popt2)
'''
G1_c = np.array(G1_c, dtype=np.float)
popt3, pconv = curve_fit(g1_func, tau, G1_c, [0.002, 1, 6000, 1000], maxfev = 20000000)
print('popt1_convective', popt3)
'''
y1 = [ g1_func(t, popt1[0], popt1[1], popt1[2], popt1[3]) for t in tau ]
y2 = [ g1_func(t, popt2[0], popt2[1], popt2[2], popt2[3]) for t in tau ]
#y3 = [ g1_func(t, popt3[0], popt3[1], popt3[2], popt3[3]) for t in tau ]

plt.plot(np.log10(tau),G1, label = 'overall_mc')
plt.plot(np.log10(tau), G1_d, label = 'diffusive_mc')
plt.plot(np.log10(tau), G1_c, label = 'convective_mc')

plt.plot(np.log10(tau), y1, label = 'overall_fit')
#plt.plot(np.log10(tau), y2, label = 'diffusive_fit')
#plt.plot(np.log10(tau), y3, label = 'convective_fit')
plt.legend()
plt.show()
'''
plt.figure(3)
x = [20]
y = [0]
n = 31
for i in range(len(simul_results[n])):
	p = simul_results[n][i]['position']
	x.append(p.x)
	y.append(p.y)

plt.plot(x,y,'ro-')
plt.show()

plt.figure(4)
r_list = []
for n in range(len(simul_results)):
	for i in range(len(simul_results[n])):
		r = simul_results[n][i]['r']
		r_list.append(r)
plt.hist(r_list, bins = 5, normed = 1)
plt.xlabel('r')
plt.ylabel('frequency')
plt.show()

plt.figure(4)
scattering_cnt = []
for n in range(len(simul_results)):
	scattering_cnt.append(len(simul_results[n]))
print(np.mean(np.array(scattering_cnt)))
print(np.std(np.array(scattering_cnt)))
plt.hist(scattering_cnt,bins = 20, normed=1)
plt.xlabel('number of scattering events')
plt.ylabel('frequency')
plt.show()


'''
