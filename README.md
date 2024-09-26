# health
sns.countplot(data = nig , x ='SleepHours', palette = 'Spectral')
![image](https://github.com/user-attachments/assets/d3df45c7-ce31-41bb-939c-5ff58b5812a3)

sns.boxplot(data = nig , x = 'Sex' , y = 'SleepHours' , hue = 'Sex')
![image](https://github.com/user-attachments/assets/e298339e-f1e2-43aa-8eab-6cb6b02fc316)

nig2 = nig [ ['HeightInMeters','WeightInKilograms']].dropna()
nig2.head()
![image](https://github.com/user-attachments/assets/22c65065-a034-40fb-89f4-0fc352627103)

model.cluster_centers_
![image](https://github.com/user-attachments/assets/7c50d080-1b46-4bdf-8cd7-aeed192ee0f9)

plt.figure(figsize=[7,6])
sns.scatterplot(data = nig2 , x = 'HeightInMeters' , y='WeightInKilograms')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1] ,color='r')
![image](https://github.com/user-attachments/assets/a03aae6f-8d09-47cd-9b82-4e4013fa2f45)

model.labels_
plt.figure(figsize=[7,6])
sns.scatterplot(data = nig2 ,  x = 'HeightInMeters' , y='WeightInKilograms'
               ,hue = model.labels_,palette = 'Set1')
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1] ,color='k')
![image](https://github.com/user-attachments/assets/52245095-8327-4b16-b957-31fca501e22e)

sns.pairplot(nig)
![image](https://github.com/user-attachments/assets/c947edb3-c457-4062-bcc8-ff4ae86dc351)

g=sns.FacetGrid(data=nig,col='Sex')
g.map(sns.histplot,'BMI',bins=10)
![image](https://github.com/user-attachments/assets/3cb5ce63-390d-43c6-a2ba-9046784dbec1)

att = nig[['HeightInMeters','WeightInKilograms']]
label = nig['Sex']

att_train , att_test, class_train , class_test = train_test_split(att,label,random_state = 0,train_size=0.7)

scaler = StandardScaler()
scaler.fit(att_train)

att_train[['HeightInMeters','WeightInKilograms']] = scaler.transform(att_train)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(att_train,class_train)

model.score(scaler.transform(att_test), class_test)
![image](https://github.com/user-attachments/assets/18696adb-d894-41c1-9742-c7f016a95ef8)

result = pd.concat([att_test,class_test], axis = 1)
result['Gender'] = model.predict(scaler.transform(att_test))
result 
![image](https://github.com/user-attachments/assets/7e28f4c2-71a7-4236-983f-48014c409f55)

sns.histplot(nig['SleepHours'], bins = 25 )
![image](https://github.com/user-attachments/assets/375ae879-c984-4050-95df-bb93af99fb8c)

sns.countplot(data=nig,x='Sex')
![image](https://github.com/user-attachments/assets/bdc4196f-bb71-4be7-a0d3-15acc452508f)

lb=nig['GeneralHealth'].unique()
data=nig['GeneralHealth'].value_counts()
plt.pie(data,labels=lb,autopct="%.1f%%")
plt.show()
![image](https://github.com/user-attachments/assets/c548c454-a88f-4560-a5fa-337e6fe84b4c)

lb1=nig['SmokerStatus'].unique()
data=nig['SmokerStatus'].value_counts()
plt.pie(data,labels=lb1,autopct="%.1f%%")
plt.show()
![image](https://github.com/user-attachments/assets/0f938503-84e7-4d52-ab31-5cfee24fe29b)

plt.figure(figsize=(15, 6))
ax = sns.lineplot(x="GeneralHealth", y="SleepHours", data=nig)
ax2 = ax.twinx()
ax2 = sns.lineplot(x="GeneralHealth", y="SleepHours", data=nig, color='red')
![image](https://github.com/user-attachments/assets/3855e6c7-bcc0-4f1f-9618-425288e35973)
