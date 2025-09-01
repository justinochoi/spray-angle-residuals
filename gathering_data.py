from pybaseball import statcast 
from pybaseball import cache 

cache.enable() 

statcast_23 = statcast(start_dt='2023-03-30', end_dt='2023-10-01') 
statcast_24 = statcast(start_dt='2024-03-28', end_dt='2024-09-30')  

statcast_23.to_parquet('/Users/justinchoi/BaseballData/statcast_23.parquet') 
statcast_24.to_parquet('/Users/justinchoi/BaseballData/statcast_24.parquet')
