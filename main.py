from prep import data_reader, preprocess
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

tr_te_pct = 0.2
tr_val_pct = 0.2
combined_df = data_reader.read_combined()

Y = combined_df['Binding']
combined_df = combined_df.drop('Binding', axis=1)

# apply min-max normalization to each column
combined_df = combined_df.apply(preprocess.min_max_normalize, axis=0)

X = combined_df

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=tr_te_pct, random_state=None)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=tr_val_pct, random_state=None)

with tf.device('/device:GPU:0'):
    model = RandomForestClassifier(random_state=0, n_estimators=50, warm_start=True, n_jobs=-1)

nsamples, nx, ny, nz = x_train.shape
xx = x_train.reshape((nsamples, nx*ny))
model.fit(xx, y_train)
nsamples, nx, ny, nz = x_test.shape
xx = x_test.reshape((nsamples, nx*ny))
y_pred = model.predict(xx)

accuracy_score(y_test, y_pred.round())
