#~ PREREQUISITE:
#~ conda install pandas
from pandas import read_csv

df = read_csv('Filosofi2017_carreaux_200m_mart.csv')

#~ l'individu est identifie de manière exclusive par son 'Idcar_200m'
#~ DICTIONNAIRE DES VARIABLES
#~ https://www.insee.fr/fr/statistiques/6215138?sommaire=6215217#dictionnaire

#~ ======================================================================
df = df.drop(columns=['I_est_200', 'Idcar_1km', 'I_est_1km', 'Idcar_nat', 'Groupe', 'lcog_geo'])

dfNum = df.select_dtypes(float).copy(deep=True)

#~ dfNum.fillna(-99999, inplace=True)

#~ PREREQUISITE:
#~ conda install openpyxl

#~ df.to_excel('data.xlsx')

#~ ======================================================================
#~ CLUSTERING
#~ PREREQUISITE:
#~ conda install scikit-learn
from sklearn.cluster import KMeans

K, sumOfSquaredDist = range(10, 1, -1), []
for nclusters in K:
	kmeans = KMeans(n_clusters=nclusters, init='k-means++', n_init=10, max_iter=300)
	kmeans = kmeans.fit(dfNum)
	#~ df[f'clusters_{nclusters}'] = kmeans.predict(dfNum)
	sumOfSquaredDist.append(kmeans.inertia_)

#~ -----
import matplotlib.pyplot as plt

plt.plot(K, sumOfSquaredDist, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#~ ======================================================================
#~ CLUSTERING
#~ from scipy.cluster.hierarchy import dendrogram, linkage

#~ plt.figure(figsize=(2.8 * 8.26, 1.2 * 8.26))
#~ plt.title(f'Classification par dendrogramme (nclusters={nclusters})', fontsize=24)
#~ d = dendrogram(Z, orientation='top', show_contracted=True, no_plot=False)
#~ Z = linkage(dfNum, method='ward', metric='euclidean')
#~ nclusters = len(list(set(d['color_list']))) - 1
#~ plt.grid(False)
#~ plt.show()

#~ ======================================================================

#~ caract, nIndivParCaract = np.unique(df.X5, return_counts=True)

#~ plt.figure(figsize=(8.26, 8.26))
#~ plt.title('Effectifs selon X5')
#~ plt.pie(nIndivParCaract, labels=caract, autopct='%1.1f%%')
#~ plt.show()
