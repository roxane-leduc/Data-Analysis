#~ ======================================================================
#~ CREATION JEU DE DONNEES
import numpy as np
from numpy.random import choice, rand, randint, seed
from pandas import concat, DataFrame

def foo():
	row = {
		'X1': randint(18, 100),
		'X2': 36 * rand(),
		'X3': 3 * rand(),
		'X4': 127 * rand(),
		'X5': choice(['A', 'A', 'B', 'C', 'C', 'C'])
	}
	row.update({ 'Y': 13 * row['X2'] + 7 * row[f'X4'] + 31 + rand() })
	return row

seed(0) # Je force l'initialisation du générateur pseudo-aléatoire
nrows = 13
df = DataFrame(data=[foo() for _ in range(nrows)])

#~ df = df.head(14)
nomsVariables = df.columns[:-1].tolist()

print(f'{len(df)} individus, {len(df.columns)-1} variables {nomsVariables}')

#~ ======================================================================
import matplotlib.pyplot as plt

caract, nIndivParCaract = np.unique(df.X5, return_counts=True)

plt.figure(figsize=(8.26, 8.26))
plt.title('Effectifs selon X5')
plt.pie(nIndivParCaract, labels=caract, autopct='%1.1f%%')
plt.show()

#~ ======================================================================
plt.figure(figsize=(8.26, 8.26))
plt.title('Effectifs selon X1 (age)')

n, bins, _ = plt.hist(df.X1, bins=None, density=False, cumulative=False,
	range=(0,100), facecolor='gray', edgecolor='black', hatch='', 
	orientation='vertical', histtype='bar', align='mid', rwidth=0.95)
plt.show()

#~ ======================================================================
#~ EXTRACTION DES SEULES VARIABLES *CONTINUES*
#~ dfnum = df.select_dtypes(include='number')
df2 = df.select_dtypes(include='float')

#~ -----
#~ PUIS EXTRACTION DES SEULES VARIABLES EXPLICATIVES
df2 = df2.drop(columns=['Y'])

#~ -----
#~ CENTRER-REDUIRE LES VARIABLES
df2 = (df2 - df2.mean()) / df2.std(ddof=0)

#~ ======================================================================
#~ CREATION D'UNE MATRICE NUMPY
pop = df2.to_numpy()

#~ ======================================================================
#~ CENTRE DE GRAVITE - INDIVIDU MOYEN
G = np.mean(pop, axis=0)

#~ -----
#~ CALCUL INERTIE TOTALE
n = len(df2) # nb d'individus
inertieTotale = sum( ( (df2 - df2.mean())**2 ).sum() ) / n
print(f'inertieTotale (1) = {inertieTotale}')

#~ -----
#~ AUTRE CALCUL INERTIE TOTALE
inertieTotale = np.sum( np.sum( (pop-G)**2, axis=1 ) ) / n
print(f'inertieTotale (2) = {inertieTotale}')

#~ -----
#~ AUTRE CALCUL INERTIE TOTALE
inertieTotale = np.sum( df2.var(ddof=0) ) # LE ddof=0 EST OBLIGATOIRE !
print(f'inertieTotale (3) = {inertieTotale}')

#~ -----
#~ AUTRE CALCUL INERTIE TOTALE
inertieTotale = np.sum( pop.var(axis=0) )
print(f'inertieTotale (4) = {inertieTotale}')

#~ ======================================================================
R = df2.corr().to_numpy()
valPP, vectPP = np.linalg.eig(R)

#~ -----
#~ FIRST EIGEN VALUE
i = np.argmax(valPP)
valPP1, vectPP1 = valPP[i], vectPP[:,i]

df3 = df2.copy()
df3['pc1'] = np.sum(df2 * vectPP1, axis=1)

#~ -----
#~ SECOND EIGEN VALUE
i = valPP.argsort()[-2]
valPP2, vectPP2 = valPP[i], vectPP[:,i]

df3['pc2'] = np.sum(df2 * vectPP2, axis=1)

#~ ON VERIFIE QUE pc1 ET pc2 SONT ORTHOGONAUX
df3.corr()
