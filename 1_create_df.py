"""Levanta todos los archivos csv de la carpeta logs, crea dos dataframes y los guarda en pickles"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

LOG_PATH = "logs"
ROTURAS_PATH = "Lista roturas.xlsx"
#===== NAME FORMAT =======================================================
# MX_HY_inertial.csv MX_HYY_inertial.csv MX_HY_cont.csv MX_HYY_cont.csv
#=========================================================================

roturas = pd.read_excel(ROTURAS_PATH)


def largo_rotura(roturas, n_helice, df_cont, nombre_columna, nombre_excel):
    df_cont[nombre_columna] = int(
        roturas[roturas["Número de hélice"] == int(n_helice.strip("ab"))][nombre_excel]
    )


n = 0
df_cont = pd.DataFrame()
df_inertial = pd.DataFrame()


for root, dirs, files in os.walk(LOG_PATH, topdown=False):
    for name in tqdm(files):
        if "cont" in name or "inertial" in name:
            # read data
            df_aux = pd.read_csv(os.path.join(root, name))
            
            #get motor number, propeller number
            n_motor = name[1]
            n_helice = name.strip("_cont.csv").strip("_inertial.csv").strip("ab")
            n_helice = n_helice[4:]     #MX_HY_cont o MX_HYY_cont
            
            # drop first and last "cut" seconds
            cut = 4
            fs = 222
            df_aux.drop(df_aux.index[:cut*fs], inplace=True)
            df_aux.drop(df_aux.index[-cut*fs:], inplace=True)
            
            # info
            largo_rotura(roturas, n_helice, df_aux, "prof1", "Profundidad 1 [mm]")
            largo_rotura(roturas, n_helice, df_aux, "prof2", "Profundidad 2 [mm]")
            df_aux["motor"] = int(n_motor)
            df_aux["tipo"] = str(roturas[roturas["Número de hélice"] == int(n_helice)]["Tipo de rotura"].to_string(index=False))
            df_aux["test"] = root
            df_aux["dataset"] = n
            
            # add to corresponding dataframe
            if "cont" in name:
                df_cont = pd.concat([df_cont, df_aux], ignore_index=True)
            elif "inertial" in name:
                df_inertial = pd.concat([df_inertial, df_aux], ignore_index=True)
                
            n += 1

print(df_inertial.head())
print(df_inertial.tail())
print(df_inertial.describe())
print(df_inertial.info())

# sns.heatmap(df_inertial.corr())
# sns.pairplot(df_inertial, hue="dataset")
# sns.scatterplot(data=df_inertial, x="prof2", y="accelerometer_m_s2_3", hue="prof1")

print(df_inertial["dataset"].value_counts())

# Rename Time column to timestamp
df_cont.rename(columns={"Time": "timestamp"}, inplace=True)
# Convert timestamp column to datetime
df_cont["timestamp"] = pd.to_datetime(df_cont["timestamp"])
df_inertial["timestamp"] = pd.to_datetime(df_inertial["timestamp"])

# Save dataframes as pickles
df_cont.to_pickle("df_cont.pickle")
df_inertial.to_pickle("df_inertial.pickle")
