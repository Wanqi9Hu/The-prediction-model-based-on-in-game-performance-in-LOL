
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tqdm import tqdm

data = pd.read_csv('temp.csv')
categorical = ['firstBlood', 'firstTower', 'firstBaron', 'firstDragon', 'firstRiftHerald']

data_categorical = data[categorical]


plt.figure(figsize=(18, 12))

cont_cnt = 0
for feature in categorical:
    cont_cnt += 1
    plt.subplot(2, 3, cont_cnt)
    plt.grid()
    counts = data[feature].value_counts()
    labels = []
    for index in counts.index:
        labels.append(f'{feature} {index}')
    plt.pie(counts, autopct='%1.1f%%')
    plt.legend(labels=labels)
    plt.title('Distribution - ' + feature)
    plt.xlabel(f'Value - {feature}')
    plt.ylabel('Count #')

plt.suptitle('Plot #2 Distribution - Categorical Features')
plt.show()


data_encoded = data
data_encoded[categorical] = data_encoded[categorical].astype(object)
data_encoded = pd.get_dummies(data_encoded)

y = to_categorical(data_encoded['win'].astype(int), num_classes=2)

X = data_encoded.drop('win', axis=1).values
X_df = pd.DataFrame(X)
X_df.columns = data_encoded.drop('win', axis=1).columns
X = scale(data_encoded.drop('win', axis=1).values)
feature_train, feature_test, target_train, target_test = train_test_split(X, y, test_size=0.2)

input_dim = feature_train.shape[1]

model = keras.models.Sequential([
    Input(input_dim),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(2)
])

model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.001),
              loss='mse', metrics=['categorical_accuracy'])

train_history = model.fit(feature_train, target_train, epochs=50, verbose=1)

plt.figure(figsize=(8, 6))
plt.plot(train_history.history['categorical_accuracy'], label='Accuracy')
plt.xlabel('Iteration #')
plt.title('Learning Curve - MLPC Model')
plt.show()

evaluate_result = model.evaluate(feature_test, target_test, verbose=1)
print('Accuracy:', evaluate_result[1])

print('Model Summary:')
print(model.summary())
print('Saving...')
model.save('model/model_MLPC')


model.evaluate(feature_test, target_test, verbose=1)

classes = np.array([0, 1])

y_pred = np.argmax(model.predict(feature_test), axis=-1)
y_true = np.vectorize(lambda x: x[1], signature='(n)->()')(target_test)
con_mat = tf.math.confusion_matrix(y_true, y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix - MLPC Model')
sns.heatmap(con_mat_df, square=True, annot=True)

plt.show()


cross_validation_record = []

def train_model():
    feature_train, feature_test, target_train, target_test = train_test_split(X, y, test_size=0.2)
    model = keras.models.Sequential([
        Input(input_dim),
        Dense(64, activation='relu'),
        #         Dropout(0.1),
        Dense(64, activation='relu'),
        #         Dropout(0.1),
        Dense(2)
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.005),
                  loss='mse', metrics=['categorical_accuracy'])
    train_history = model.fit(feature_train, target_train, epochs=50, verbose=1)
    evaluate_result = model.evaluate(feature_test, target_test, verbose=1)
    cross_validation_record.append(evaluate_result[1])
    return cross_validation_record

with tf.device('/cpu:0'):
    for _ in tqdm(range(50)):
        train_model()

plt.figure(figsize=(8, 6))
sns.kdeplot(cross_validation_record)
plt.title('Cross Validation - MLPC Model')
plt.xlabel('Accuracy')
plt.show()


res_df = pd.DataFrame(pd.Series(cross_validation_record).describe())
res_df.columns = ['Value']
print(res_df.T)

plt.figure(figsize=(8, 6))

def generate_avg_entry(data):
    data_copy = data.copy().astype(float)
    stat = data_copy.describe()
    data_copy.iloc[0] = stat.loc['mean', :]
    for item in categorical:
        data_copy.loc[0, item] = round(stat[item]['mean'])
    return data_copy.iloc[0]


def generate_cont_test_data(avg_entry, data, feature):
    cnt = 1000
    data_copy = data.copy().drop('win', axis=1)
    data_copy.iloc[0] = avg_entry
    data_copy[categorical] = data_copy[categorical].astype(int).astype(object)
    cont_stat = data.describe()
    max_val = cont_stat[feature]['75%']
    min_val = cont_stat[feature]['25%']
    interval = np.linspace(min_val, max_val, cnt)
    avg_entry_encoded = pd.get_dummies(data_copy).iloc[0]

    res = pd.DataFrame(np.tile(avg_entry_encoded, cnt).reshape(cnt, -1))
    res.columns = data_encoded.drop('win', axis=1).columns
    res[feature] = interval
    return res.values


avg_entry = generate_avg_entry(data)

convert_to_score = np.vectorize(lambda x: (x[1] - x[0]) / (abs(x[0]) + abs(x[1])), signature='(n)->()')
plt.figure(figsize=(18, 10))


age_var_data = generate_cont_test_data(avg_entry, data, 'firstBaron')
age_var_df = pd.DataFrame(age_var_data)
age_var_df.columns = data_encoded.drop('win', axis=1).columns

feature_cat_dict = dict()

for feature in categorical:
    feature_cat_dict[feature] = []
    for val in np.array(data[feature].value_counts().index):
        feature_cat_dict[feature].append(f'{feature}_{val}')

for feature in feature_cat_dict:
    feature_cat_dict[feature] = sorted(feature_cat_dict[feature])

plot_cnt = 0
plt.figure(figsize=(14, 14))
