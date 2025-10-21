# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 02:02:23 2025

@author: lenovo
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
import numpy as np
import math
import random

import tdgl
from tdgl.geometry import box, circle
class defect:
    def __init__(self,d_inf):
        self.xc=d_inf[1]
        self.yc=d_inf[2]
        self.rx=d_inf[3]
        self.ry=d_inf[4]
        self.theta=d_inf[5]
        self.epsilon=d_inf[6]
        
def make_epsilon(defects):
    def epsilon(r):
        x,y=r
        for d in defects:
            if (((x-d.xc)*math.cos(d.theta)+(y-d.yc)*math.sin(d.theta))/d.rx)**2+(((y-d.yc)*math.cos(d.theta)-(x-d.xc)*math.sin(d.theta))/d.ry)**2<=1:
                return d.epsilon
        return 1
    return epsilon

def compute_v(defects,current,field,index,seed=None):
    #output_file=os.path.join(tempdir.name,f"test{index}.h5")
    #options.output_file=output_file
    zero_field_solution = tdgl.solve(
        device,
        options,
        applied_vector_potential=field,
        disorder_epsilon=make_epsilon(defects),
        terminal_currents=dict(source=current, drain=-current),
        seed_solution=seed,
    )
    dynamics = zero_field_solution.dynamics
    v_ave=dynamics.mean_voltage(tmin=100)
    
    return v_ave,zero_field_solution

length_units = "um"
# Material parameters
xi = 0.5
london_lambda = 2
d = 0.05
layer = tdgl.Layer(coherence_length=xi, london_lambda=london_lambda, thickness=d, gamma=1)
# Device geometry
total_width = 8
total_length = 16
film = (
    tdgl.Polygon("film", points=box(total_width, total_length))
    .resample(401)
    .buffer(0)
)
# Current terminals
source = (
    tdgl.Polygon("source", points=box(1.1 * total_width, total_length / 100))
    .translate(dy=total_length / 2)
)
drain = source.scale(yfact=-1).set_name("drain")
#  Voltage measurement points
probe_points = [(0, total_length / 2.5), (0, -total_length / 2.5)]
device = tdgl.Device(
    "square",
    layer=layer,
    film=film,
 #   holes=[round_hole, square_hole],
    terminals=[source, drain],
    probe_points=probe_points,
    length_units=length_units,
)

device.make_mesh(max_edge_length=xi, smooth=100)
options = tdgl.SolverOptions(
    # Allow some time to equilibrate before saving data.
    skip_time=400,
    solve_time=400,
    field_units = "mT",
    current_units="uA",
    save_every=1000,
    
)

def compute_jc(defects,fields,max_iterate,threshold,j_max,j_min):
    seed=None
    left=j_min
    right=j_max
    device.make_mesh(max_edge_length=xi, smooth=100)
    fine_mesh=False
    mid=(left+right)/2
    for i in range(max_iterate):
        last_mid=mid
        mid=(left+right)/2
        mid_v,seed=compute_v(defects,mid,fields,i,seed=None)
        if abs(mid_v-threshold)<=0.01 and fine_mesh==False:
            fine_mesh=True
            device.make_mesh(max_edge_length=xi/2,smooth=100)
            mid_v,seed=compute_v(defects,last_mid,fields,i,seed=None)
        if abs(mid_v-threshold)<=0.005 or i==max_iterate-1:
            return mid,mid_v
        elif mid_v>threshold:
            right=mid
        elif mid_v<threshold:
            left=mid   
    return mid,mid_v

random.seed(1)
np.random.seed(1)
input_array=[]
target_array=[]
threshold=0.02
max_iterate=7
field=0.1
num=200
max_defects=50
for i in range(num):
    j_c=0
    v_c=1.0
    n_defects=random.randint(0,max_defects)
    defects=[]
    array=np.zeros((max_defects,7))
    for k in range(n_defects):
        xc=random.uniform(-4.0,4.0)
        yc=random.uniform(-8.0,8.0)
        defects.append(defect((1,xc,yc,1.0,1.0,0,-1)))
        array[k][0]=1
        array[k][1]=xc
        array[k][2]=yc
        array[k][3]=1.0
        array[k][4]=1.0
        array[k][5]=0
        array[k][6]=-1
    j_c,v_c=compute_jc(defects,field,max_iterate,threshold,15,0)
    input_array.append(array)
    target_array.append(j_c)
    file=open('progress1.out','a')
    file.write(str(i))
    file.write('\n')
    file.close()
input_torch=torch.from_numpy(np.block(input_array))
target_torch=torch.from_numpy(np.block(target_array))
torch.save({'inputs':input_torch,'targets':target_torch},'data1m.pth')

