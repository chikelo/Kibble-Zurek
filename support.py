# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:35:31 2019

@author: Eric
"""

import os

import pandas as pd
import numpy as np
import scipy.optimize as spo
import decimal

# Convert from dense to pandas array
idx = pd.IndexSlice

def get_names(directory,ext='.dat'):

    base_names = [os.path.join(root,os.path.splitext(file)[0]) 
              for root, dirs, files in os.walk(directory) 
              for file in files 
              if file.endswith(ext)]
    base_names = list(np.sort(np.array(base_names)))
    
    return base_names 

def dense2multiindex(trj_dat):
    no_of_col = trj_dat.shape[1]
    no_of_part = round((no_of_col-1)/2)
    
    times = trj_dat[0].values
    indexes = np.arange(0,no_of_part)
    
    index = pd.MultiIndex.from_product([times,indexes],names=('time','id'))
    
    trj = pd.DataFrame(
        {"x":trj_dat.iloc[:,1::2].values.flatten(),
         "y":trj_dat.iloc[:,2::2].values.flatten()},index=index)
    return trj
# load .dat and convert
def load_dat(name):
    date_parser = lambda time : pd.to_datetime(
        float(time)+2*3600, unit="s", origin=pd.Timestamp('1904-01-01'))
    
    trj_dat = pd.read_csv(name+'.dat',sep="\t",
                header=None, parse_dates=[0], date_parser=date_parser)
    
    trj = dense2multiindex(trj_dat)

    trj["frame"] = trj.index.get_level_values("time")
    #trj["name"] = os.path.split(name)[-1]
    trj["particle"] = trj.index.get_level_values("id")
    
    trj = trj.set_index(["frame","particle"]) #(["name","frame","particle"])
    
    return trj

def load_data(name):
    
    trj = pd.read_csv(name+".dat", sep = "\t", index_col = 0)
    trj = trj.filter(["frame", "particle", "x", "y"])
    trj = trj.set_index(["frame", "particle"])
    return trj

def from_px_to_um(trj,px_size):
    trj.x = trj.x*px_size # microns per pixel
    trj.y = trj.y*px_size # microns per pixel
    return trj

def check_N_particles(trj):
    
    particle_list = trj.index.get_level_values("particle").unique().to_list()
    N_frames = len(trj.index.get_level_values("frame").unique().to_list())
    
    if len(particle_list) > 50:
        for particle in particle_list:
            if len(trj.loc[idx[:,particle],:])<N_frames:
                trj = trj.drop(index = particle, level = "particle")    
    
    trj = trj.sort_index()
    
    return trj

def get_center(trj):
    
    def calc_R(xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((trj.x.values-xc)**2 + (trj.y.values-yc)**2)

    def f_2(c):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()    

    center_estimate = 0, 0
    center, ir = spo.leastsq(f_2, center_estimate)
    return center

def recenter(trj):
    
    center = get_center(trj.loc[idx[0,:],:].copy())
    trj.x+=-center[0]
    trj.y+=-center[1]
    return trj

def get_polar_coordinates(trj):
    
    trj["r"] = np.sqrt(trj.x**2 + trj.y**2)
    trj["theta"] = np.arctan2(trj.y,trj.x)
    
    return trj

def find_final_frame(trj, final_radius):
    
    last_frame = 0
    digits = abs(decimal.Decimal(str(final_radius)).as_tuple().exponent)
    
    for frame in trj.index.get_level_values("frame").unique().to_list():
        
        if round(trj.loc[idx[frame,:],"r"].mean(),digits) == final_radius:
            last_frame = frame
            break
    
    return last_frame
    
def sort_particles_in_circle(trj):
    
    # The new id will replace the old particle id. They are the same, except the new one is sorted by increasing angle. 
        
    new_id = trj.loc[idx[:,:],:].sort_values("theta").filter(["theta"])
    new_id["id"] = range(0,len(new_id))
    new_id = new_id.sort_index()
    
    trj = trj.reset_index()
    trj.particle = new_id.id.values
    trj = trj.set_index(["frame", "particle"]).sort_index()
    
    return trj.sort_index()

def set_particle_positions(trj):
    
    trj["position"] = np.NaN
    r_mean = trj.r.mean()

    for particle in trj.index.get_level_values("particle").unique().to_list():
        
        if trj.loc[idx[:,particle],"r"].values > r_mean:
            
            trj.loc[idx[:,particle],"position"] = 1
            
        elif trj.loc[idx[:,particle],"r"].values < r_mean:
            
            trj.loc[idx[:,particle],"position"] = -1
            
        else:
            trj.loc[idx[:,particle],"position"] = 0
            
    return trj

def configure_domains(trj):
    
    trj["domain"] = np.NaN
    trj.loc[idx[:,0],"domain"] = 0

    for particle in trj.index.get_level_values("particle").unique().to_list()[1:]:
        
        if trj.loc[idx[:,particle],"position"].values != trj.loc[idx[:,particle-1],"position"].values:
            trj.loc[idx[:,particle],"domain"] = trj.loc[idx[:,particle-1],"domain"].values
            
        else:
            trj.loc[idx[:,particle],"domain"] = trj.loc[idx[:,particle-1],"domain"].values+1
        
        last_particle = trj.index.get_level_values("particle").unique().to_list()[-1]
    
        if trj.loc[idx[:,last_particle],"position"].values != trj.loc[idx[:,0],"position"].values:
            domain_merge_id = trj[trj.domain.values == trj.loc[idx[:,last_particle],"domain"].values].index.get_level_values("particle").values
            trj.loc[idx[:,domain_merge_id], "domain"] = [0]*len(domain_merge_id)
            
        #else:
        #    trj.loc[idx[:,domain_merge_id], "domain"] = [0]*len(domain_merge_id)
        
    return trj

def obtain_domain_walls(trj):

    last_particle = trj.index.get_level_values("particle").unique().to_list()[-1]

    trj["domain_walls"] = trj.position.diff()
    trj.domain_walls[0] = trj.position.values[0]-trj.position.values[last_particle]
    
    return trj

def find_domain_head(trj):
    
    particle_list = trj[trj.domain == 0].index.get_level_values("particle").unique().to_list()
    head_particle = 0
    if len(particle_list) > 1:
        for i in np.linspace(0,len(particle_list)-1,len(particle_list),dtype=int):
            if particle_list[i+1] - particle_list[i] != 1:
                head_particle = particle_list[i+1]
                break

    return head_particle

def set_particle_position_multiframe(trj):
    
    trj["position"] = np.NaN
    
    for f in trj.index.get_level_values("frame").unique().to_list():
        
        r_mean = trj.loc[idx[f,:], "r"].mean()
        
        for particle in trj.index.get_level_values("particle").unique().to_list():
        
            if trj.loc[idx[f,particle],"r"] > r_mean:

                trj.loc[idx[f,particle],"position"] = 1

            elif trj.loc[idx[f,particle],"r"] < r_mean:

                trj.loc[idx[f,particle],"position"] = -1

            else:
                trj.loc[idx[f,particle],"position"] = 0
            
    return trj

def sort_particles_in_circle_multiframe(trj):
    
    all_id = pd.DataFrame()
    
    for f in trj.index.get_level_values("frame").unique().to_list():
        
        new_id = trj.loc[idx[f,:],:].sort_values("theta").filter(["theta"])
        new_id["id"] = range(0,len(new_id))
        new_id = new_id.sort_index()
        
        all_id = all_id.append(new_id, ignore_index=True)
        #trj.loc[idx[f,:], :] = trj.loc[idx[f,:], :].index.set_levels(new_id.id, level="particle")

    trj = trj.reset_index()
    trj.particle = all_id.id #new_id.id.values
    trj = trj.set_index(["frame", "particle"]).sort_index()
    
    return trj.sort_index()

def find_domain_walls(trj):

    trj["domain_walls"] = np.NaN

    for f, trj_sub in trj.groupby("frame"):

        last_particle = trj.index.get_level_values("particle").unique().to_list()[-1]
        trj.loc[idx[f,:], "domain_walls"] = trj.loc[idx[f,:], "position"].diff()
        trj.loc[idx[f,0], "domain_walls"] = trj.loc[idx[f,0], "position"]-trj.loc[idx[f,last_particle], "position"]
        
    return trj

def find_lowest_final_frame(trj):
    
    min_frame = 20000000000000000

    for test, t_sub in trj.groupby("test_num"):
        
        if len(t_sub["frame"])<min_frame:
            min_frame = len(t_sub["frame"])
                     
    return min_frame
