import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model as load_tf_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
from joblib import load

# Load pre-trained models
nn_model = tf.keras.models.load_model('../models/NN_model_20240227.h5')
xgb_model = joblib.load('../models/best_xgb_model_20240227.pkl')
svr_model = joblib.load('../models/best_svr_model.pkl')

# Load data from CSV
data = pd.read_csv('../data/20230930_aquasat_L2_C2.csv')

# Assuming the target variable is named 'secchi'
X = data.drop('secchi', axis=1)
y = data['secchi']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6, random_state=42)


# Normalize data using training set
scaler = StandardScaler().fit(X_train)
X_train_norm = scaler.transform(X_train)
X_val_norm = scaler.transform(X_val)
X_test_norm = scaler.transform(X_test)

# Save normalization parameters
joblib.dump(scaler, '../data/normalization_scaler_20240227.pkl')

# Load models and make predictions
#nn_model = load_tf_model('NN_model.h5')
#xgb_model = joblib.load('best_xgb_model.pkl')
#svr_model = joblib.load('best_svr_model.pkl')


nn_preds_train = nn_model.predict(X_train_norm)
xgb_preds_train = xgb_model.predict(X_train_norm)
svr_preds_train = svr_model.predict(X_train_norm)

nn_preds_val = nn_model.predict(X_val_norm)
xgb_preds_val = xgb_model.predict(X_val_norm)
svr_preds_val = svr_model.predict(X_val_norm)

nn_preds_test = nn_model.predict(X_test_norm)
xgb_preds_test = xgb_model.predict(X_test_norm)
svr_preds_test = svr_model.predict(X_test_norm)




p_nn = np.concatenate([nn_preds_train, nn_preds_val, nn_preds_test]).reshape(-1, 1)
p_xgb = np.concatenate([xgb_preds_train, xgb_preds_val, xgb_preds_test]).reshape(-1, 1)
p_svr = np.concatenate([svr_preds_train, svr_preds_val, svr_preds_test]).reshape(-1, 1)
p_secchi = np.concatenate([y_train, y_val, y_test]).reshape(-1, 1)
#p_secchi = y.ravel().reshape(-1, 1)

# Create new dataset
ensemble_data = pd.DataFrame({
    'nn_preds': p_nn.ravel(),
    'xgb_preds': p_xgb.ravel(),
    'svr_preds': p_svr.ravel(),
    'secchi_true': p_secchi.ravel()
})


# Set up early stopping and model checkpointing
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = keras.callbacks.ModelCheckpoint('../models/best_ens_model_20240227.h5', monitor='val_loss', save_best_only=True)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir='./logs')

# Split ensemble data
X_ensemble = ensemble_data.drop('secchi_true', axis=1)
y_ensemble = ensemble_data['secchi_true']


r2 = []
for i in np.arange(20):
    X_train_ens, X_temp_ens, y_train_ens, y_temp_ens = train_test_split(X_ensemble, y_ensemble, test_size=0.5)#, random_state=42)
    X_val_ens, X_test_ens, y_val_ens, y_test_ens = train_test_split(X_temp_ens, y_temp_ens, test_size=0.6)#, random_state=42)


    # Normalize ensemble data
    scaler_ens = StandardScaler().fit(X_train_ens)
    X_train_ens_norm = scaler_ens.transform(X_train_ens)
    X_val_ens_norm = scaler_ens.transform(X_val_ens)
    X_test_ens_norm = scaler_ens.transform(X_test_ens)


    # Save ensemble normalization parameters
    #joblib.dump(scaler_ens, '../data/ensemble_normalization_scaler.pkl')





    # Train ensemble neural network
    #ensemble_nn = Sequential([
    #    Dense(45, activation='relu', input_shape=(X_train_ens_norm.shape[1],)),

    #    Dense(448, activation='relu'),
    #    Dense(1)
    #])
    #ensemble_nn.compile(optimizer='adam', loss='mean_squared_error')
    #ensemble_nn.fit(X_train_ens_norm, y_train_ens, validation_data=(X_val_ens_norm, y_val_ens), epochs=500, verbose=1, callbacks=[early_stopping, checkpoint, tensorboard_callback])

    ensemble_nn = load('../models/ensemble_best_model_20240227.pkl')
    # Evaluate ensemble model
    y_pred_ens = ensemble_nn.predict(X_test_ens_norm)
    r2.append(r2_score(y_test_ens, y_pred_ens))
    print(f"Ensemble Model R^2: {r2}")


df = pd.DataFrame({'r2': r2})
df.to_csv('../data/ensemble_r2_20240227.csv', index=False)
print(f"Ensemble Model R^2: {np.mean(r2)}")
print(f"Ensemble Model R^2: {np.std(r2)}")
# Save ensemble model
ensemble_nn.save('ensemble_nn_model_20240227.h5')