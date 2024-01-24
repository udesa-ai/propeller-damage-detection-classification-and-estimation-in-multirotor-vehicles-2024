import pandas as pd
from tqdm import tqdm
import numpy as np

df_inertial = pd.read_pickle("df_inertial.pickle")
df_cont = pd.read_pickle("df_cont.pickle")


# Joining inertial and cont, equal timestamps
df=pd.merge(df_inertial, df_cont, left_index=True, right_index=True)
df=df.drop(columns=["timestamp_y", 'prof1_y','prof2_y', 'tipo_y', 'test_y', 'dataset_y', 'motor_y'])

# Rename columns
df.rename(
    columns={
        "accelerometer_m_s2_1": "acc_x",
        "accelerometer_m_s2_2": "acc_y",
        "accelerometer_m_s2_3": "acc_z",
        "gyro_rad_1": "gyro_x",
        "gyro_rad_2": "gyro_y",
        "gyro_rad_3": "gyro_z",
        "control_1": "cmd_pitch",
        "control_2": "cmd_roll",
        "control_3": "cmd_yaw",
        "control_4": "cmd_thrust",
        "dataset_x": "dataset",
        'timestamp_x': 'timestamp', 
        'prof1_x':'prof1', 
        'prof2_x':'prof2', 
        'tipo_x':'tipo',
        'test_x':'test', 
        'motor_x':'motor',
    },
    inplace=True,
)


# Save interior
df_interior = df[df['test'].str.contains("Garage")]
df_interior.to_pickle("df_int.pickle")

# Save exterior
df_exterior = df[df['test'].str.contains("Exterior")]
df_exterior.to_pickle("df_ext.pickle")

print(df.head())
print(df.tail())
print(df.sample())
print(df.columns)
print(df.drop_duplicates(subset=['dataset'])[["dataset", "prof1", "prof2", "tipo", "test"]])
