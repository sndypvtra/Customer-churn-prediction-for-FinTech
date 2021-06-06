# Mengimpor library yang diperlukan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mengimpor dataset
df = pd.read_csv("data_input/data_fintech.csv") 

# Ringkasan data
ringkasan = df.describe()
tipe_data = df.dtypes

# Merevisi kolom num_screens
df['screen_list'] = df.screen_list.astype(str) + ','
df['num_screens'] = df.screen_list.str.count(',')
df.drop(columns=['numscreens'], inplace=True)

# Cek kolom hour
df.hour[1] # terdapat sepasi sebelum angka 0 pada jam maka akan dilakukan slicing
df.hour = df.hour.str.slice(1,3).astype(int)

# Mendefinisikan variabel khusus numerik
df_numerik = df.drop(columns=['user','first_open','screen_list',
                                      'enrolled_date'], inplace=False)

# Membuat histogram
sns.set()
plt.suptitle('Histogram Data Numerik')
for i in range(0, df_numerik.shape[1]):
    plt.subplot(3,3,i+1)
    figure = plt.gca()
    figure.set_title(df_numerik.columns.values[i])
    jumlah_bin = np.size(df_numerik.iloc[:,i].unique())
    plt.hist(df_numerik.iloc[:,i], bins=jumlah_bin)

# Membuat correlation bar(mencari korelasi antara variable dependent dengan semua variable independent) plot dan correlation matrix(mencari korelasi antar variable independent)
korelasi = df_numerik.drop(columns=['enrolled'], inplace=False).corrwith(df_numerik.enrolled)
korelasi.plot.bar(title='Korelasi variabel terhadap keputusan Enrolled')
# Insight : 1. jumlah num_screens dan jumlah pelanggan yang memainkan minigame (independent) berkorelasi positif terhadap enrolled(dependent) tapi tidak terlalu besar yaitu 0.3/1.0 dan 0.1/1.0.     
#           2. semakin semakin banyak fitur yang ada dan semakin banyak screen yang pelanggan explore terlepas dari fitur itu gratis atau premium maka akan semakin besar potensi pelanggan untuk enrolled/berlangganan.         
#           3. serta dapat terlihat bahwa pelanggan yang menggunakan fitur premium dan pelanggan yang semakin tua justru berkorelasi negatif/ tidak berlangganan, mungkin saja bisa dikatakan bahwa usia yang semakin tua akan lebih mencoba fitur premium saja dan tidak untuk berlangganan.
#           4. secara umum num_screen(independent) yang akan dijadikan prediktor utama dalam kasus ini.
# Conclusion of insight : dari sini kita tau bahwa pelanggan yang memutuskan berlangganan itu lebih sering explore pada aplikasi, maka ide nya adalah bisa saja pada bagian development aplikasi/websitenya harus bisa memberi performa yang optimal dan kesan semenarik mungkin, sehingga akan sangat berpotensi pada pelanggan untuk enrolled.
    
matriks_korelasi = df_numerik.drop(columns=['enrolled'], inplace=False).corr()
sns.heatmap(matriks_korelasi, cmap='Blues')

mask = np.zeros_like(matriks_korelasi, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Membuat correlation matrix dnegan heatmap 
ax = plt.axes()
cmap_ku = sns.diverging_palette(200, 0, as_cmap=True)
sns.heatmap(matriks_korelasi, cmap=cmap_ku, mask=mask, 
            linewidths=0.5, center=0, square=True)
ax = plt.suptitle('Correlation Matrix')

# FEATURE ENGINEERING
# proses parsing
from dateutil import parser
df.first_open = [parser.parse(i) for i in df.first_open]
df.enrolled_date = [parser.parse(i) if isinstance(i, str) else i for i in df.enrolled_date]
df['selisih'] = (df.enrolled_date - df.first_open).astype('timedelta64[h]')

# Membuat plot histogram dataku.selisih
plt.hist(df.selisih.dropna(), range=[0,200])
plt.suptitle('Selisih waktu antara enrolled dengan first open')
plt.show()

# Memfilter nilai selisih > 48 jam
df.loc[df.selisih>48, 'enrolled'] = 0

# Mengimpor data top_screens 
top_screens = pd.read_csv('data_input/top_screens.csv')
# Data diatas merupakan data yang memili

top_screens = np.array(top_screens.loc[:,'top_screens'])

# Membuat cadangan data
df2 = df.copy()

# Membuat kolom untuk setiap top_screens
for layar in top_screens:
    df2[layar] = df2.screen_list.str.contains(layar).astype(int)
    
for layar in top_screens:
    df2['screen_list'] = df2.screen_list.str.replace(layar+',', '')

# Item non top_screens di screen_list
df2['lainnya'] = df2.screen_list.str.count(',')

top_screens.sort()

# Proses penggabungan beberapa screen yang sama (Funneling)
layar_loan = ['Loan',
              'Loan2',
              'Loan3',
              'Loan4']
df2['jumlah_loan'] = df2[layar_loan].sum(axis=1) # axis=1 artinya menghitung jumlah item per baris
df2.drop(columns=layar_loan, inplace=True) # menghilangkan semua kolom yg menjadi list savings_screen

layar_saving = ['Saving1',
                'Saving2',
                'Saving2Amount',
                'Saving4',
                'Saving5',
                'Saving6',
                'Saving7',
                'Saving8',
                'Saving9',
                'Saving10']
df2['jumlah_saving'] = df2[layar_saving].sum(axis=1) 
df2.drop(columns=layar_saving, inplace=True) 

layar_credit = ['Credit1',
                'Credit2',
                'Credit3',
                'Credit3Container',
                'Credit3Dashboard']
df2["jumlah_kredit"] = df2[layar_credit].sum(axis=1)
df2.drop(columns=layar_credit, inplace=True)

layar_cc = ['CC1',
            'CC1Category',
            'CC3']
df2['jumlah_cc'] = df2[layar_cc].sum(axis=1)
df2.drop(columns=layar_cc, inplace=True)

# Mendefinisikan var dependen
var_enrolled = np.array(df2['enrolled'])

# Menghilangkan beberapa kolom yang redundan
df2.drop(columns = ['first_open', 'screen_list','enrolled',
                        'enrolled_date', 'selisih'], inplace=True)

# Membagi menjadi training dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2, var_enrolled, 
                                                    test_size=0.2,
                                                    random_state=111)
                                                    
# Menyimpan user ID untuk training dan test set
train_id = np.array(X_train['user'])
test_id = np.array(X_test['user'])

# Menghilangkan kolom user di X_train dan X_test
X_train.drop(columns=['user'], inplace=True)
X_test.drop(columns=['user'], inplace=True)

# Merubah X_train dan X_test menjadi numpy array (test set sudah berbentuk array ajdi tidak perlu)
X_train = np.array(X_train)
X_test = np.array(X_test)

# Preprocessing Standardization (Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Menghilangkan variabel kosong
X_train = np.delete(X_train, 27, 1)
X_test = np.delete(X_test, 27, 1)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='liblinear',
                                penalty = 'l1')
classifier.fit(X_train, y_train)

# Memprediksi test set
y_pred = classifier.predict(X_test)

# Mengevaluasi model dengan confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# Menggunakan accuracy_score
evaluasi = accuracy_score(y_test, y_pred)
print('Akurasi:{:.2f}'.format(evaluasi*100))

# Menggunakan seaborn untuk CM
cm_label = pd.DataFrame(cm, columns = np.unique(y_test),
                        index = np.unique(y_test))
cm_label.index.name = 'Aktual'
cm_label.columns.name = 'Prediksi'
sns.heatmap(cm_label, annot=True, cmap='Reds', fmt='g')

# Validasi dengan 10-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train,
                             cv=10)
accuracies.mean()
accuracies.std()
print('Akurasi Regresi Logistik = {:.2f}% +/- {:.2f}%'.format(accuracies.mean()*100, accuracies.std()*100))

# Menggabungkan semuanya
y_pred_series = pd.Series(y_test).rename('asli', inplace=True)
hasil_akhir = pd.concat([y_pred_series, pd.DataFrame(test_id)], axis=1).dropna()
hasil_akhir['prediksi'] = y_pred
hasil_akhir.rename(columns={0:'user'}, inplace = True)
hasil_akhir = hasil_akhir[['user','asli','prediksi']].reset_index(drop=True)
