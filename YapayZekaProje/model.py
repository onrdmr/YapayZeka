# %%
# kütüphanelerin import edilmesi
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import pingouin as pg

from scipy.stats import mode

# %%
# veri setinin okunması
df = pd.read_csv('osuDataset.csv')
print(df.shape)
df['mode'].value_counts()

# %%
# veri setinin gereksiz verilerden arınması
df = df.query("mode == 'osu' and ranked == 1")
print(df.shape)
df['mode'].value_counts()

# %%
# veri seti özellikleri hakkında bilgiler
df.info(verbose=True, null_counts=True)

# %%
# kullanılacak özelliklerin belirlenmesi
y = df['difficulty_rating']
x_ = df[['hit_length','cs','drain','ar','accuracy','bpm']]
x_.describe()

# %%
# normalizasyon
scaler = MinMaxScaler()
x = scaler.fit_transform(x_)
pd.DataFrame(x).describe()

# %%
y.head(10)

# %%
plt.show()
plt.rcParams["figure.figsize"] = (15,5.5)

# %%
# 
plt.style.use('fivethirtyeight')
y_ = y.to_numpy()
plt.scatter(range(len(y_)), y_)#np.sort(y_))
plt.xticks(range(0, len(y_),int(len(y_)/20)))
plt.title("Ground Truth Diffuculty")

# %%
plt.style.use('fivethirtyeight')
plt.hist(y_,bins=30)
plt.title("Difficulty Rate Histogram")

# %%
np.max(y_)

# %%
plt.style.use('fivethirtyeight')
hit_lenght = x[:,0]
plt.xticks(range(0, len(hit_lenght),int(len(hit_lenght)/20)))
plt.scatter(range(len(hit_lenght)), x[:,0])
plt.title("hit length")
plt.show()

# %%
plt.style.use('fivethirtyeight')
cs = x[:,1]
plt.xticks(range(0, len(cs),int(len(cs)/20)))
plt.scatter(range(len(cs)), cs)
plt.title("Circle Size")
plt.show()

# %%
plt.style.use('fivethirtyeight')
drain = x[:,2]
plt.xticks(range(0, len(drain),int(len(drain)/20)))
plt.scatter(range(len(drain)), drain)
plt.title("drain")
plt.show()

# %%
plt.style.use('fivethirtyeight')
ar = x[:,3]
plt.xticks(range(0, len(ar),int(len(ar)/20)))
plt.scatter(range(len(ar)), ar)
plt.title("Approach Rate")
plt.show()

# %%
plt.style.use('fivethirtyeight')
acc = x[:,4]
plt.xticks(range(0, len(acc),int(len(acc)/20)))
plt.scatter(range(len(acc)), acc)
plt.title("accuracy")
plt.show()

# %%
plt.style.use('fivethirtyeight')
bpm = x[:,5]
plt.xticks(range(0, len(bpm),int(len(bpm)/20)))
plt.scatter(range(len(bpm)), bpm)
plt.title("bpm")
plt.show()

# %% [markdown]
# # Relations

# %% [markdown]
# ## Grupların Anova İle İncelenmesi

# %% [markdown]
# Grupları sınıflandırmak arasında anlamlı fark var mı F-testi ve ANOVA incelenmesi

# %%
plt.style.use('fivethirtyeight')
plt.scatter( hit_lenght,y_)#.astype(int))

# %%
pd.DataFrame(np.c_[hit_lenght,y_.astype(int)],columns=["hit_lenght","difficulty"]).head()

# %%
anova1_data = pd.DataFrame(np.c_[hit_lenght,y_.astype(int)],columns=["hit_lenght","difficulty"])
pg.anova(data=anova1_data,dv='hit_lenght',between='difficulty')

# %% [markdown]
# Difficult değerleri arasında p değeri 0.05 den küçük olduğu için anlamlı fark vardır yani bu grupların veriyi etkilediği söylenebilir. Tüm gruplar kendi içinde etkilimi söylemek için t test uygulanır. Bunun için pairwise_tukey test kullanılır ve gruplar kendi içinde incelenir.

# %%
pg.pairwise_tukey(data=anova1_data,dv='hit_lenght',between='difficulty').head(50)

# %% [markdown]
# Tukey testinden uzak olan gruplar arasında daha anlamlı fark vardı bu ordinal veri olmasından dolayıdır. Uzak verilerden anlamlı bilgi çıkabilir.

# %%
plt.scatter( cs, y_ )
plt.title("Difficulty-Circle Size Relation")
plt.xlabel("Circle Size")
plt.ylabel("Difficulty")
plt.show()

# %%
anova2_data = pd.DataFrame(np.c_[cs,y_.astype(int)],columns=["circle size","difficulty"])
pg.anova(data=anova2_data, dv='circle size', between='difficulty')

# %% [markdown]
# Difficult değerleri arasında p değeri 0.05 den küçük olduğu için anlamlı fark vardır yani bu grupların veriyi etkilediği söylenebilir. Tüm gruplar kendi içinde etkilimi söylemek için t test uygulanır. Bunun için pairwise_tukey test kullanılır ve gruplar kendi içinde incelenir.

# %%
pg.pairwise_tukey(data=anova2_data,dv='circle size',between='difficulty')

# %%
plt.scatter( drain, y_ )
plt.title("Difficulty-Drain Relation")
plt.xlabel("Drain")
plt.ylabel("Difficulty")
plt.show()

# %%
anova3_data = pd.DataFrame(np.c_[drain, y_.astype(int)], columns=["drain","difficulty"])
pg.anova(data=anova3_data, dv='drain', between='difficulty')

# %%
pg.pairwise_tukey(data=anova3_data, dv='drain',between='difficulty')

# %% [markdown]
# # Anova Devamı

# %%
plt.scatter( ar, y_ )
plt.title("Difficulty-Approach Rate Relation")
plt.xlabel("Approach Rate")
plt.ylabel("Difficulty")
plt.show()

# %%
anova4_data = pd.DataFrame(np.c_[ar, y_.astype(int)],columns=["approach rate","difficulty"])
pg.anova(data=anova4_data, dv='approach rate', between='difficulty')

# %%
pg.pairwise_tukey(data=anova4_data, dv='approach rate',between='difficulty')

# %%
plt.scatter( bpm, y_ )
plt.title("Difficulty-bpm Relation")
plt.xlabel("bpm")
plt.ylabel("difficulty")
plt.show()

# %%
anova5_data = pd.DataFrame(np.c_[ar, y_.astype(int)],columns=["bpm","difficulty"])
pg.anova(data=anova5_data, dv='bpm', between='difficulty')

# %%
pg.pairwise_tukey(data=anova5_data, dv='bpm',between='difficulty')

# %%
plt.scatter( acc, y_ )
plt.title("Difficulty Drain Relation")
plt.xlabel("accuracy")
plt.ylabel("difficult")
plt.show()

# %%
anova6_data = pd.DataFrame(np.c_[ar, y_.astype(int)],columns=["accuracy","difficulty"])
pg.anova(data=anova6_data, dv='accuracy', between='difficulty')

# %%
pg.pairwise_tukey(data=anova6_data, dv='accuracy',between='difficulty')

# %% [markdown]
# One Way ANOVA incelemesi sonucu gereksiz verilerin atılmasına karar verilmiştir.

# %%
pd.DataFrame(x, columns=['hit_length','cs','drain','ar','accuracy','bpm']).boxplot()

# %% [markdown]
# 12-11-10-9-0 zorluk değerlerinden anlamlı veri çıkmamaktadır bu yüzden yeterli veri toplanana kadar bu veri atılır.  

# %%
difficult_count = [np.count_nonzero(anova3_data['difficulty'] == i) for i in range(int(np.max(anova3_data['difficulty'])))]
plt.bar(range(len(difficult_count)),difficult_count,color ='maroon',width = 0.4)

# %%
difficult_count

# %% [markdown]
# # Gereksiz Verinin Atılması

# %%
# uç noktadaki verilerin temizlenmesi
dataset = np.c_[x,y]
reduced_ds=dataset[dataset[:,6] >= 1]
reduced_ds=reduced_ds[reduced_ds[:,6] < 9]
np.min(reduced_ds[:,6]), np.max(reduced_ds[:,6])

# %%
x = reduced_ds[:,:6]
y = reduced_ds[:,6]

# %%
anova_data = pd.DataFrame(np.c_[x[:,0], y.astype(int)],columns=["hit_length","difficulty"])
pg.anova(data=anova_data, dv='hit_length', between='difficulty')

# %%
pg.pairwise_tukey(data=anova_data, dv='hit_length',between='difficulty').head()

# %% [markdown]
# # Hazır Modeller İle Deneme

# %%
# hazır sklearn algoritmaları ile kfold kullanarak modelleme
models = [LinearRegression(), KNeighborsRegressor(), DecisionTreeRegressor(), SVR()]
results = []

for model in models:
    results.append(cross_val_score(model, x, y, cv = 10).sum() / 10)

results

# %% [markdown]
# # 0 dan fonksiyonlar

# %% [markdown]
# KNN classification

# %%
# öklid uzaklığı
def eucledian(p1,p2):
    dist = np.sqrt(np.sum((p1-p2)**2))
    return dist



# knn ile regresyon fonksiyonu
def predict(x_train, y , x_input, k):
    op_labels = []
     
    # Sınıflandırılacak veriler için döngü
    for item in x_input: 
         
        # uzaklıkları saklayan dizi
        point_dist = []
         
        # eğitim verileri için döngü
        for j in range(len(x_train)): 
            distances = eucledian(np.array(x_train[j,:]) , item) 
            # uzaklığı hesapla
            point_dist.append(distances) 
        point_dist = np.array(point_dist) 
         
        # dizi sıralama
        # en yakın k komşuyu al
        dist = np.argsort(point_dist)[:k] 
         
        # komşuların ortalamasını al
        values = y[dist]
        num = 0
        for i in values:
            num = num + i
            
        # çıktıya ekle
        op_labels.append(num / k)
 
    return op_labels

# %%
# kfold ile knn regresyon testi
kf = KFold(n_splits=10)
res = []

for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_pred = predict(x_train, y_train, x_test, 7)

        res.append(r2_score(y_test, y_pred))

knnres = np.array(res).sum() / 10
knnres

# %%
(len(x_train),len(x_test)) , (len(y_train),len(y_test))

# %% [markdown]
# KNN Broadcast Regresyon

# %%
X_train = x_train
X_test = x_test
knn_y_train = y_train
knn_y_test = y_test

# %%
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

knn_X_train = (X_train - mu ) / sigma

knn_X_test = (X_test - mu ) / sigma

mu_y = np.mean(y_train, 0)
sigma_y = np.std(y_train, 0, ddof = 0)

knn_y_train = (y_train - mu_y ) / sigma_y

# %%
(len(knn_X_train),len(knn_X_test)) , (len(knn_y_train),len(knn_y_test))

# %%
import time

start = time.process_time()

k_list = [x for x in range(1,50,1)]

distance = np.sqrt(((knn_X_train[:, :, None] - knn_X_test[:, :, None].T) ** 2).sum(1))

sorted_distance = np.argsort(distance, axis = 0)

# %%


def knn(X_train, X_test, y_train, y_test, sorted_distance, k):
    y_pred = np.zeros(y_test.shape)
    for row in range(len(X_test)):
        
        y_pred[row] = y_train[sorted_distance[:,row][:k]].mean() * sigma_y + mu_y

    RMSE = np.sqrt(np.mean((y_test - y_pred)**2))
    return RMSE




# %%
rmse_list = []

# %%
# her k değeri için rmse değerlerini listeye kaydet
for i in k_list:
    rmse_list.append(knn(knn_X_train,knn_X_test,knn_y_train,knn_y_test,sorted_distance,i))
    
print(time.process_time() - start)

# %%
# en iyi k değerini görmek için rmse değerlerini karşılaştır
plt.plot(k_list, rmse_list)
plt.xlabel("K values")
plt.ylabel("RMSE")


# %%
# en iyi k değerini bul
min_rmse_k_value = k_list[rmse_list.index(min(rmse_list))]

# en iyi rmse değerini bul
optimal_RMSE = knn(knn_X_train,knn_X_test,knn_y_train,knn_y_test,sorted_distance,min_rmse_k_value)
rmse = optimal_RMSE / (np.max(y)-np.min(y))

# %%
print("Accuracy : %" , 100*(1-rmse))

# %%
# bizim knn algoritmamız ile sklearn karşılaştırması
plt.style.use('fivethirtyeight')
plt.figure(dpi=100)

plt.barh('sklearn', results[1])
plt.barh('bizimki', knnres)

plt.xlabel('r2-score')
plt.title('KNN Regression r2-score')

plt.savefig('fig/KNN.png', bbox_inches='tight')

plt.show()

# %%
min_rmse_k_value

# %% [markdown]
# Lineer Ağırlıklı Regresyon (Kendimiz Yazdık)

# %%
# lineer regresyon class
class LR():

    # hiper parametre girdisi
    def __init__(self, alpha, iterations):
        self.alpha = alpha
        self.iterations = iterations

    # model eğitme fonksiyonu
    def fit(self, X, Y):
        # örnek sayısı, özellik sayısı
        self.m, self.n = X.shape

        # değerlerin atanması
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y

        # gradient descent
        for i in range(self.iterations):
            self.update_weights()

        return self

    # w ve b değerlerini yenileyen fonksiyon
    def update_weights(self):
        Y_pred = self.predict(self.X)

        # gradyan hesapla
        dW = - (2 * (self.X.T).dot(self.Y - Y_pred)) / self.m
        db = - 2 * np.sum(self.Y - Y_pred) / self.m

        # w ve b değerini yenile
        self.W = self.W - self.alpha * dW
        self.b = self.b - self.alpha * db

        return self

    # h(x) fonksiyonu
    def predict(self, X):

        return X.dot(self.W) + self.b


# %%
# modelin kfold kullanılarak test edilmesi ve r2 score çıktısı

kf = KFold(n_splits=10)
model = LR(iterations = 1000, alpha = 0.01)
res = []

for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(x_train, y_train)
        
        y_pred = model.predict(x_test)

        res.append(r2_score(y_test, y_pred))

lrres = np.array(res).sum() / 10
lrres

# %%
# bizim ve sklearn lineer regresyon karlışaltırması
plt.style.use('fivethirtyeight')
plt.figure(dpi=100)

plt.barh('sklearn', results[0])
plt.barh('bizimki', lrres)

plt.title('Linear Regression r2-score')
plt.xlabel('r2-score')

plt.savefig('fig/LR.png', bbox_inches='tight')

plt.show()

# %%
# tüm algoritmaların karşılaştırılması
plt.style.use('fivethirtyeight')
plt.figure(dpi=100)

plt.bar('LR', results[0], label='sklearn Linear Regression')
plt.bar('KNN', results[1], label='sklearn KNeighbors Regression')
plt.bar('DTR', results[2], label='sklearn Decision Tree Regression')
plt.bar('SVR', results[3], label='sklearn Support Vector Regression')
plt.bar('KNN-2', knnres, label='Bizim KNN Regression Algoritmamız')
plt.bar('LR-2', lrres, label='Bizim Linear Regression Algoritmamız')

plt.title('Algoritma Karşılaştırması')
plt.ylabel('r2-score')
plt.xlabel('Algoritma')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

plt.savefig('fig/r2scores.png', bbox_inches='tight')

plt.show()


