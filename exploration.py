import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

statcast_23 = pd.read_parquet('/Users/justinchoi/BaseballData/statcast_23.parquet') 
statcast_24 = pd.read_parquet('/Users/justinchoi/BaseballData/statcast_24.parquet')

bbe_23 = statcast_23[statcast_23['description'] == 'hit_into_play'].reset_index(drop=True)
bbe_24 = statcast_24[statcast_24['description'] == 'hit_into_play'].reset_index(drop=True)

cols = ['release_speed','plate_x','plate_z','pfx_x','pfx_z','hc_x','hc_y','launch_angle','launch_speed']

bbe_23.dropna(subset=cols, inplace=True)
bbe_24.dropna(subset=cols, inplace=True) 

# first need to calculate spray angle 
# adjustments from openWAR package 
def calc_spray_angle(data):

    data['hc_x'] = data['hc_x'] - 125 
    data['hc_y'] = 199 - data['hc_y'] 
    data['spray_angle'] = (np.arctan2(data['hc_y'], data['hc_x']) * 180 / np.pi) - 90  

    return data 

# flip x-coordinates for lefty batters 
def adjust_vars(data): 
    for col in ['plate_x', 'pfx_x', 'spray_angle']: 
        data[col] = np.where(data['stand'] == 'L', data[col].mul(-1), data[col])

    return data 

bbe_23 = calc_spray_angle(bbe_23) 
bbe_23 = adjust_vars(bbe_23)
bbe_24 = calc_spray_angle(bbe_24) 
bbe_24 = adjust_vars(bbe_24)

sns.histplot(x=bbe_23['spray_angle']) 
sns.histplot(x=bbe_24['spray_angle']) 
# hitters generally end up towards their pull side 

# which variables correlate with spray angle? 
# vertical/horizontal plate location, swing length?
fig, ax = plt.subplots() 
plt.hexbin(x=bbe_23['plate_x'], y=bbe_23['plate_z'], C=bbe_23['spray_angle'], gridsize=(7,5)) 
plt.axvline(x=0, linestyle='--', color='red')
plt.axhline(y=2.5, linestyle='--', color='red')
plt.colorbar(label="Spray Angle (deg.)")
plt.suptitle("Hitter Spray Angle by Pitch Location", fontsize=16)
ax.set_title("Positive = Pulled / Negative = Oppo")
ax.set_xlabel("Horizontal Plate Location $(ft.)$") 
ax.set_ylabel("Vertical Plate Location $(ft.)$")
ax.text(2, 0.5, "Outside to Hitter", ha="right", va="center", 
        bbox=dict(boxstyle="rarrow, pad=0.3", fc='lightblue', ec='steelblue', lw=2))
ax.text(-2, 0.5, "Inside to Hitter", ha="left", va="center", 
        bbox=dict(boxstyle="larrow, pad=0.3", fc='lightblue', ec='steelblue', lw=2))
# low pitches are pulled more often 
# inside pitches are pulled more often 

# now trying out movement 
fig, ax = plt.subplots() 
plt.hexbin(x=bbe_23['pfx_x'], y=bbe_23['pfx_z'], C=bbe_23['spray_angle'], gridsize=10) 
plt.axvline(x=0, linestyle='--', color='red')
plt.axhline(y=0, linestyle='--', color='red')
plt.colorbar(label="Spray Angle (deg.)")
plt.suptitle("Hitter Spray Angle by Pitch Movement", fontsize=16)
ax.set_title("Positive = Pulled / Negative = Oppo")
ax.set_xlabel("Horizontal Movement $(ft.)$") 
ax.set_ylabel("Vertical Movement $(ft.)$")
ax.text(2, -1.5, "Outside to Hitter", ha="right", va="center", 
        bbox=dict(boxstyle="rarrow, pad=0.3", fc='lightblue', ec='steelblue', lw=2))
ax.text(-2, -1.5, "Inside to Hitter", ha="left", va="center", 
        bbox=dict(boxstyle="larrow, pad=0.3", fc='lightblue', ec='steelblue', lw=2))
# pitches with negative IVB are pulled more often 
# horizontal break doesn't seem to have much impact
