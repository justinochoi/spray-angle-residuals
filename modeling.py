import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier 
import xgboost as xgb 
import optuna 

# model spray angle using the following:
# release_speed, pfx_x, pfx_z, plate_x, plate_z 
# cross-validation by splitting players into mutually exclusive folds 

statcast_23 = pd.read_parquet('/Users/justinchoi/BaseballData/statcast_23.parquet') 
statcast_24 = pd.read_parquet('/Users/justinchoi/BaseballData/statcast_24.parquet')

bbe_23 = statcast_23[statcast_23['description'] == 'hit_into_play'].reset_index(drop=True)
bbe_24 = statcast_24[statcast_24['description'] == 'hit_into_play'].reset_index(drop=True)

cols = ['release_speed','plate_x','plate_z','pfx_x','pfx_z','hc_x','hc_y','launch_angle','launch_speed']

bbe_23.dropna(subset=cols, inplace=True)
bbe_24.dropna(subset=cols, inplace=True) 

X = bbe_23[['release_speed','pfx_x','pfx_z','plate_x','plate_z']] 
y = bbe_23['spray_angle'] 
batters = bbe_23['batter'] 

def xgb_objective(trial): 
    params = {
        "objective": "reg:squarederror",
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),    
        "monotone_constraints": {
            'release_speed': 0, 
            'pfx_x': 0, 
            'plate_z': -1, 
            'pfx_z': -1, 
            'plate_x': -1
        }  
    }

    cv = GroupKFold(n_splits=5, shuffle=True, random_state=76)
    cv_scores = [] 

    for train_idx, val_idx in cv.split(X, y, groups=batters): 
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx] 
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx] 

        model = xgb.XGBRegressor(**params) 
        model.fit(X_train, y_train)  
        preds = model.predict(X_val) 
        rmse = np.sqrt(mean_squared_error(y_val, preds)) 
        cv_scores.append(rmse) 
    
    return np.mean(cv_scores)

spray_angle_study = optuna.create_study(direction='minimize') 
spray_angle_study.optimize(xgb_objective, n_trials=20) 

best_params = spray_angle_study.best_params 

spray_angle_mod = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=500, 
    learning_rate= best_params['learning_rate'], 
    max_depth=best_params['max_depth'], 
    subsample=best_params['subsample'], 
    colsample_bytree=best_params['colsample_bytree'], 
    min_child_weight=best_params['min_child_weight'],
    monotone_constraints={
        'release_speed': 0, 
        'pfx_x': 0, 
        'plate_z': -1, 
        'pfx_z': -1, 
        'plate_x': -1
    }
)
spray_angle_mod.fit(X, y) 
bbe_23['pred_spray'] = spray_angle_mod.predict(X) 
bbe_23['spray_resid'] = bbe_23['spray_angle'] - bbe_23['pred_spray']

# now we'll build three versions of xwoba 
# 1. launch_speed + launch_angle (currently used in Savant) 
# 2. launch_speed + launch_angle + spray_angle 
# 3. launch_speed + launch_angle + spray_resid 

# turn every out into a field_out
def simplify_events(data): 

    data['events'] = np.where(
        data['events'] == 'single', 'single', np.where(
            data['events'] == 'double', 'double', np.where(
                data['events'] == 'triple', 'triple', np.where(
                    data['events'] == 'home_run', 'home_run', 'field_out'
                )
            )
        )
    )

    return data 

bbe_23 = simplify_events(bbe_23) 
bbe_24 = simplify_events(bbe_24)

X_basic = bbe_23[['launch_speed','launch_angle']] 
X_spray = bbe_23[['launch_speed','launch_angle','spray_angle']] 
X_spray_resid = bbe_23[['launch_speed','launch_angle','spray_resid']] 
y_events = bbe_23['events']

xwoba_basic = KNeighborsClassifier(n_neighbors=10)
xwoba_basic.model_name = 'basic' 
xwoba_spray = KNeighborsClassifier(n_neighbors=10)
xwoba_spray.model_name = 'spray'
xwoba_spray_resid= KNeighborsClassifier(n_neighbors=10)
xwoba_spray_resid.model_name = 'spray_resid'

xwoba_basic.fit(X_basic, y_events) 
xwoba_spray.fit(X_spray, y_events) 
xwoba_spray_resid.fit(X_spray_resid, y_events) 

# lexicographic order: double, field_out, home_run, single, triple 
def add_woba_preds(data, model):
    
    name = model.model_name
    feats = model.feature_names_in_
    preds = model.predict_proba(data[feats]) 
    for i, event in enumerate(['double','field_out','home_run','single','triple']): 
        data[f'{name}_{event}_prob'] = preds[:,i]

    return data

bbe_23 = add_woba_preds(bbe_23, xwoba_basic)
bbe_23 = add_woba_preds(bbe_23, xwoba_spray)
bbe_23 = add_woba_preds(bbe_23, xwoba_spray_resid) 

def calculate_xwobacon(data): 
    # linear weights from fangraphs 
    single_val = .883 
    double_val = 1.244 
    triple_val = 1.569 
    home_run_val = 2.004 

    for model in ['basic','spray','spray_resid']: 
        single_ev = data[f'{model}_single_prob'] * single_val 
        double_ev = data[f'{model}_double_prob'] * double_val 
        triple_ev = data[f'{model}_triple_prob'] * triple_val 
        home_run_ev = data[f'{model}_home_run_prob'] * home_run_val
        data[f'{model}_xwobacon'] = single_ev + double_ev + triple_ev + home_run_ev 

    return data 

bbe_23 = calculate_xwobacon(bbe_23) 

# now the million dollar question 
# does using spray angle residual lead to a more predicitive xwobacon? 
xwobacon_23 = (
    bbe_23
    .groupby('batter')
    .agg(
        n = ('batter', 'size'), 
        wobacon_23 = ('estimated_woba_using_speedangle','mean'), 
        basic_xwobacon = ('basic_xwobacon', 'mean'), 
        spray_xwobacon = ('spray_xwobacon', 'mean'), 
        spray_resid_xwobacon = ('spray_resid_xwobacon', 'mean')
    )
    .query('n >= 10')
    .reset_index()
)

wobacon_24 = (
    bbe_24
    .groupby('batter')
    .agg(
        n = ('batter','size'), 
        wobacon_24 = ('estimated_woba_using_speedangle', 'mean')
    )
    .query('n >= 10')
    .reset_index() 
)

wobacon_comp = pd.merge(xwobacon_23, wobacon_24, how = 'inner', on = 'batter')
wobacon_comp[['basic_xwobacon','spray_xwobacon','spray_resid_xwobacon','wobacon_23','wobacon_24']].corr()  

np.sqrt(mean_squared_error(wobacon_comp['wobacon_24'], wobacon_comp['spray_resid_xwobacon'], sample_weight=wobacon_comp['n_x'])) 
np.sqrt(mean_squared_error(wobacon_comp['wobacon_24'], wobacon_comp['spray_xwobacon'], sample_weight=wobacon_comp['n_x'])) 
np.sqrt(mean_squared_error(wobacon_comp['wobacon_24'], wobacon_comp['basic_xwobacon'], sample_weight=wobacon_comp['n_x'])) 
np.sqrt(mean_squared_error(wobacon_comp['wobacon_24'], wobacon_comp['wobacon_23'], sample_weight=wobacon_comp['n_x'])) 
# note: you do worse without the monotonic constraints 

