import numpy as np
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class kmeans:

    def __init__(self,k=2,iteration=100):
        self.k = k
        self.iteration = iteration
        self.centroid = []
        self.label = []
        self.wcss = 0
    
    def __calculate_distance(self,data):
        distance = []
        for i in range(self.k):
            euclid = np.linalg.norm(data-self.centroid[i,:],axis=1)
            distance.append(euclid)
        return distance

    def fit(self,data):
        #Initialize centroid
        data = np.array(data)
        data_unique = np.unique(data,axis=0)
        self.centroid = np.array(random.choices(data_unique,k=self.k))

        for _ in range(self.iteration):
            distance = self.__calculate_distance(data)
            self.label = np.argmin(distance,axis=0)

            group = defaultdict(list)

            for idx,cluster in enumerate(self.label):
                group[cluster].append(data[idx])

            new_centroid = []
            for i in range(self.k):
                mean = np.mean(group[i],axis=0)
                if np.all(np.isnan(mean)):
                    mean = np.full((self.centroid.shape[1],),np.nan)
                new_centroid.append(mean)
            
            new_centroid = np.array(new_centroid)
            if np.all(self.centroid == new_centroid):
                break
            else:
                for idx in np.argwhere(~np.isnan(new_centroid)):
                    self.centroid[idx[0],idx[1]] = new_centroid[idx[0],idx[1]]
        
        self.wcss = self.__get_wcss(data)
    
    def predict(self,data):
        data = np.array(data)
        distance = self.__calculate_distance(data)
        return np.argmin(distance,axis=0)
    
    def __get_wcss(self,data):
        data = np.array(data)
        distance = self.__calculate_distance(data)
        self.label = np.argmin(distance,axis=0)
        
        wcss_value = 0
        for i,j in enumerate(self.label):
            wcss_value += distance[j][i]**2
        
        return wcss_value

#=====================================================================#

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def ubah_ke_numerik(df,columns=[]):
    encode = LabelEncoder()
    for col in columns:
        df[col] = encode.fit_transform(df[col])
    return df

def hapus_outlier(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q3 - Q1
    
    UPPER = Q3 + 1.5*IQR
    LOWER = Q1 - 1.5*IQR
    
    df = df[(df["Premi"] >= LOWER) & (df["Premi"] <= UPPER)]
    return df

from sklearn.impute import SimpleImputer

def pre_processing_data_train(df):
    # Menangani data yang hilang atau kosong
    df['SIM'].fillna(0,inplace=True)
    df['Sudah_Asuransi'].fillna(0,inplace=True)
    df['Umur_Kendaraan'].fillna("< 1 Tahun",inplace=True)
    df['Kendaraan_Rusak'].fillna("Tidak",inplace=True)
    
    imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    df['Jenis_Kelamin'] = imputer.fit_transform(df[['Jenis_Kelamin']])
    df['Umur'] = imputer.fit_transform(df[['Umur']])
    df['Kode_Daerah'] = imputer.fit_transform(df[['Kode_Daerah']])
    df['Premi'] = imputer.fit_transform(df[['Premi']])
    df['Kanal_Penjualan'] = imputer.fit_transform(df[['Kanal_Penjualan']])
    df['Lama_Berlangganan'] = imputer.fit_transform(df[['Lama_Berlangganan']])

    # Mengubah data categorical menjadi numerik
    columns = ["Jenis_Kelamin","Umur_Kendaraan","Kendaraan_Rusak"]
    df = ubah_ke_numerik(df,columns=columns)

    # Menangani data outlier
    df = hapus_outlier(df,"Premi")

    return df


from collections import Counter
import matplotlib.pyplot as plt

def main():
    # Memuat data
    print("Sedang memuat data train....\n")
    df = pd.read_csv("https://github.com/helmy-faishal/tugas-clustering/blob/main/kendaraan_train.csv?raw=true")
    print(f"Datasets Train:\n{df}\n\n")

    # Pra-pemrosesan data
    print("Sedang melakukan pra-pemrosesan data...\n")
    df = pre_processing_data_train(df)
    selected_columns = ["Sudah_Asuransi","Kendaraan_Rusak","Umur","Kanal_Penjualan"]
    train = df[selected_columns]

    # Clustering
    print("Sedang melakukan clustering data train...\n")
    km = kmeans(k=2)
    km.fit(train)
    
    print("\n")
    for i,c in enumerate(km.centroid):
        print(f"Centroid {i} = {c}")
    print("\n")

    # Evaluasi Clustering
    print("Sedang melakukan evaluasi clustering...\n")
    list_wcss = []
    for i in range(1,6):
        km_eval = kmeans(k=i)
        km_eval.fit(train)
        list_wcss.append(km_eval.wcss)
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1,6),list_wcss)
    plt.scatter(range(1,6),list_wcss)
    plt.xticks(range(1,6))
    plt.xlabel("K Cluster")
    plt.ylabel("WCSS")
    plt.title("Elbow Method")
    plt.draw()
    

    # Prediksi data test
    print("Sedang melakukan prediksi data test...\n")
    df_test = pd.read_csv("https://github.com/helmy-faishal/tugas-clustering/blob/main/kendaraan_test.csv?raw=true")
    print(f"Datasets Test:\n{df_test}\n\n")
    
    columns = ["Jenis_Kelamin","Umur_Kendaraan","Kendaraan_Rusak"]
    df_test = ubah_ke_numerik(df_test,columns=columns)
    test = df_test[selected_columns]

    predict_test = km.predict(test)
    print(f"\nHasil prediksi:\n{Counter(predict_test)}")

    
    plt.show()
    print("\nProgram Selesai...\n")

    

main()