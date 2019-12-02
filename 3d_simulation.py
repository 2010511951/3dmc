import os
import sys
import numpy as np
import random as rd
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
	def __init__(self, theta, fac, k0):
		self.theta = theta
		self.fac = fac
		self.x = math.sin(theta)*math.cos(fac)*k0
		self.y = math.sin(theta)*math.sin(fac)*k0
		self.z = math.cos(theta)*k0


def is_inside_vessel(p, geometry, spatial_resolution):
	xn = np.int(math.floor(p.x/spatial_resolution))
	zn = np.int(math.floor(p.z/spatial_resolution))
	return geometry[xn,zn]


def generate_theta(g):
	return math.acos( 1/(2*g)*(1+g**2-((1-g**2)/(1-g+2*g*rd.random()))**2 ) )

def cal_r(p, vessel_col_num, vessel_n, h_space, skull_thickness, spatial_resolution):
	#vessel_z = (vessel_n//vessel_row_num + 1)*h_space + skull_thickness
	#vessel_x = (vessel_n-(vessel_n//vessel_row_num)*vessel_row_num)*h_space
	xn = math.floor(p.x/spatial_resolution)
	zn = math.floor(p.z/spatial_resolution)
	if vessel_n-(vessel_n//vessel_col_num)*vessel_col_num == 0:
		vessel_x = vessel_col_num*h_space/spatial_resolution
		vessel_z = (vessel_n//vessel_col_num*h_space + skull_thickness)/spatial_resolution
	else:
		vessel_x = (vessel_n-(vessel_n//vessel_col_num)*vessel_col_num)*h_space/spatial_resolution
		vessel_z = ((vessel_n//vessel_col_num + 1)*h_space + skull_thickness)/spatial_resolution

	return np.sqrt( (xn-vessel_x)**2 + (zn-vessel_z)**2 )*spatial_resolution


def cal_q_m(k_in, k_out):
	return math.sqrt((k_out.x-k_in.x)**2 + (k_out.y-k_in.y)**2 + (k_out.z-k_in.z)**2)

def is_detectable(p0,k_in, NA, n):
	x = p0.x - k_in.x/k_in.z*p0.z
	y = p0.y - k_in.y/k_in.z*p0.z
	if (x-40)**2+(y)**2 > 4 or math.sin(k_in.theta) > NA/n:
		return False
	else:
		return True

# define the output
detected_photons = []
# structure of the detected_photons
# detected_photons={photon1, photon2, photon3, ..., photon_n}
# photon=np.zeros(m,6) where m is the number of scattering events
# each colomn in photon represents: vessel_n, |q|, qy, r, L, ua
# vessel_n = 0 indicates that scattering point is outside the vessel; nonzero indicates inside the vessel
# |q| the magnitude of momentum transfer
# qy the y component of the momentum transfer
# r the radial position of the scattering point
# L path length
# ua scattering coeff.
# the 1st row of photon matrix records the L and ua before the first scattering happens, vessel_n=0, |q|=0, qy=0, r=0

config = np.load('output/config_skull_1.npy')
config = config.item()
x_max = config['x_max']	# unit: mm
z_max = config['z_max']	# unit: mm
spatial_resolution = config['spatial_resolution']	# unit: mm
geometry = config['geometry']
h_space = config['h_space']	# unit: mm
vessel_radius = config['vessel_radius'] 	# unit：mm
skull_thickness = config['skull_thickness']
vessel_col_num = math.floor( (x_max)/h_space -1) 	# number of vessels for each row along the x-axis 
vessel_row_num = math.floor( (z_max-skull_thickness)/h_space -1)	# number of vessels for each column along the z-axis 
print('geometry loaded!')
print(vessel_row_num,vessel_col_num)
print('skull_thickness: {}'.format(skull_thickness))





N = 5000000	 # number of photons to simulate
N_save=1000000		 # number of photons to save per file
ut_vessel=2.2
ut_tissue=1.002
ut_skull= 1.825	
ua_vessel=0.3	# unit: mm^-1
ua_tissue=0.002 # unit: mm^-1
ua_skull=0.025		# unit: mm^-1
g_vessel=0.9
g_tissue=0.0001
g_skull=0.94	
n_tissue = 1.4
n_skull = 1.5

NA = 1 		# detector numerical aperture

lamda = 800e-6	# unit: mm
k0 = 2*math.pi/lamda

# simulation starts
print('simulation starts')


for i in range(N):
	sys.stdout.write('photon %d\r' % (i+1))
	
	photon = []
	p = point(20, 0, 0)	# where photons are launched (x,z)
	s = -math.log(1-rd.random())/ut_skull
	p0 = point(20, 0, s)
	k_in = k(0,0, k0)        # incident light perpendicular to the x-y plane, i.e. theta = 0
	scattering_cnt = 0

	scattering_event = {'vessel_num':0, 'q_m': 0, 'q_y':0, 'r':0, 'L': s, 'ua': ua_skull, 'position': p0}
	photon.append(scattering_event)

	while True:

		vessel_n = is_inside_vessel(p0, geometry, spatial_resolution)
		g = 0
		ut = 0
		ua = 0
		q_m = 0
		q_y = 0
		r = 0
		if vessel_n>=1:	# if vessel numer is no smaller than 1, i.e. the scattering point is in vessel
			g = g_vessel
			ut = ut_vessel
			ua = ua_vessel
			theta = generate_theta(g)	# based on Henyey-Greenstein function
			fac = 2*math.pi*rd.random()
			k_out = k(theta,fac, k0)
			q_m = cal_q_m(k_in,k_out)
			q_y = k_out.y - k_in.y
			r = cal_r(p0, vessel_col_num, vessel_n, h_space, skull_thickness, spatial_resolution)

			if r > vessel_radius:
				print('vessel radius error', r)
				print(p0.x, p0.z, vessel_n, vessel_col_num)
			
			s = -math.log(1-rd.random())/ut
		elif vessel_n ==0.5:	# the scattering point is in skull
			g = g_skull
			ut = ut_skull
			us = ua_skull
			theta = generate_theta(g)	# based on Henyey-Greenstein function
			fac = 2*math.pi*rd.random()
			k_out = k(theta,fac, k0)
			s = -math.log(1-rd.random())/ut
		else:		# if vessel number if zero, i.e. the scattering point is in tissue
			g = g_tissue
			ut = ut_tissue
			ua = ua_tissue
			theta = generate_theta(g)	# based on Henyey-Greenstein function
			fac = 2*math.pi*rd.random()
			k_out = k(theta,fac, k0)
			s = -math.log(1-rd.random())/ut
			
		p = point(p.x+s*math.sin(k_out.theta)*math.cos(k_out.fac), p.y+s*math.sin(k_out.theta)*math.sin(k_out.fac), p.z+s*math.cos(k_out.theta))

		scattering_cnt += 1
		# update p0, k_in
		p0 = p
		k_in = k_out

		# update photon record
		scattering_event = {'vessel_num':vessel_n, 'q_m': q_m, 'q_y':q_y, 'r':r, 'L': s, 'ua': ua, 'position': p0}
		photon.append(scattering_event)


		if p0.x<0 or p0.x>x_max or p0.z>z_max:
			break
			continue
		else:
			if p0.z>0:
				continue
			else:	# p.z<=0, i.e. the photon has returned to the launch surface
				if is_detectable(p0, k_in, NA, n_skull):
					detected_photons.append(photon)
				break
				continue

	if i == i//N_save*N_save:
		print('\nhalfway save')
		filename = 'output/skull/thickness1mm/simulation_results_{}.npy'.format(math.ceil((i+1)/N_save+109))
		print('detected photons: {}'.format(len(detected_photons)))
		np.save(filename,detected_photons)
		detected_photons=[]

print('detected photons: {}'.format(len(detected_photons)))

if detected_photons!= None:
	filename = 'output/skull/thickness1mm/simulation_results_final.npy'
	np.save(filename, detected_photons)









