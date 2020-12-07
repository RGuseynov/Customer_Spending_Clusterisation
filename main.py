import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score


df = pd.read_csv("Mall_Customers.csv")

X = df.drop("CustomerID", axis=1)
X = pd.get_dummies(X)

mms = MinMaxScaler()
mms.fit(X)
X_transformed = mms.transform(X)

X_Female = X[X["Gender_Female"] == 1]
X_Female = X_Female.drop(["Gender_Female", "Gender_Male"], axis=1)
X_Male = X[X["Gender_Female"] == 0]
X_Male = X_Male.drop(["Gender_Female", "Gender_Male"], axis=1)


Sum_of_squared_distances = []
Sil_score = []
K = range(2,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    labels = km.predict(X)
    Sum_of_squared_distances.append(km.inertia_)
    Sil_score.append(silhouette_score(X, labels))

fig = plt.figure()
ax1 = fig.add_subplot(211)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')

ax2 = fig.add_subplot(212)
plt.plot(K, Sil_score, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()

km = KMeans(n_clusters=4)
km = km.fit(X)
labels = km.predict(X)

kmFemale = KMeans(n_clusters=4)
kmFemale = kmFemale.fit(X_Female)
labels_Female = kmFemale.predict(X_Female)

kmMale = KMeans(n_clusters=4)
kmMale = kmMale.fit(X_Male)
labels_Male = kmMale.predict(X_Male)

fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')

xs = X["Annual Income (k$)"]
ys = X["Spending Score (1-100)"]
zs = X["Age"]
c = labels

ax.scatter(xs, ys, zs, c=c)

ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.set_zlabel('Age')
ax.set_title("Both sexe")


ax2 = fig.add_subplot(223, projection='3d')

xs = X_Female["Annual Income (k$)"]
ys = X_Female["Spending Score (1-100)"]
zs = X_Female["Age"]
c = labels_Female

ax2.scatter(xs, ys, zs, c=c)

ax2.set_xlabel('Annual Income (k$)')
ax2.set_ylabel('Spending Score (1-100)')
ax2.set_zlabel('Age')
ax2.set_title("Female")


ax3 = fig.add_subplot(224, projection='3d')

xs = X_Male["Annual Income (k$)"]
ys = X_Male["Spending Score (1-100)"]
zs = X_Male["Age"]
c = labels_Male

ax3.scatter(xs, ys, zs, c=c)

ax3.set_xlabel('Annual Income (k$)')
ax3.set_ylabel('Spending Score (1-100)')
ax3.set_zlabel('Age')
ax3.set_title("Male")

plt.show()
