import bambi as bmb 
import arviz as az 
import pandas as pd 

# work in progress 
# priors are way too wide, also spray angle distribution is arguably bimodal 
# this is the most basic possible model, keeping it for future reference 

statcast_23 = pd.read_parquet('/Users/justinchoi/BaseballData/statcast_23.parquet') 
statcast_24 = pd.read_parquet('/Users/justinchoi/BaseballData/statcast_24.parquet')

bbe_23 = statcast_23[statcast_23['description'] == 'hit_into_play'].reset_index(drop=True)
bbe_24 = statcast_24[statcast_24['description'] == 'hit_into_play'].reset_index(drop=True)

cols = ['release_speed','plate_x','plate_z','pfx_x','pfx_z','hc_x','hc_y','launch_angle','launch_speed']

bbe_23.dropna(subset=cols, inplace=True)
bbe_24.dropna(subset=cols, inplace=True) 

spray_mod = bmb.Model(
    'spray_angle ~ release_speed + pfx_x + pfx_z + plate_x + plate_z', 
    data = bbe_23, family = 'gaussian'
) 
spray_mod_idata = spray_mod.fit(
    tune=500, draws=500, chains=4, cores=4, 
    random_seed=77, idata_kwargs={'log_likelihood': True}, 
    nuts_sampler='blackjax'
)

az.summary(spray_mod_idata) 
az.plot_trace(spray_mod_idata) 

spray_mod.predict(spray_mod_idata, kind='response', inplace=True) 
az.plot_ppc(spray_mod_idata) 
