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

from keras_tuner import RandomSearch

# Load pre-trained models
nn_model = tf.keras.models.load_model('../models/NN_model_20240223.h5')
xgb_model = joblib.load('../models/best_xgb_model_20240227.pkl')
svr_model = joblib.load('../models/best_svr_model.pkl')

# Load data from CSV
data = pd.read_csv('../data/20230930_aquasat_homework.csv')

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
#nn_model = load_tf_model('NN_model_20240223.h5')
#xgb_model = joblib.load('best_xgb_model_20240227.pkl')
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
#for i in np.arange(20):
X_train_ens, X_temp_ens, y_train_ens, y_temp_ens = train_test_split(X_ensemble, y_ensemble, test_size=0.5)#, random_state=42)
X_val_ens, X_test_ens, y_val_ens, y_test_ens = train_test_split(X_temp_ens, y_temp_ens, test_size=0.6)#, random_state=42)


    # Normalize ensemble data
scaler_ens = StandardScaler().fit(X_train_ens)
X_train_ens_norm = scaler_ens.transform(X_train_ens)
X_val_ens_norm = scaler_ens.transform(X_val_ens)
X_test_ens_norm = scaler_ens.transform(X_test_ens)


    # Save ensemble normalization parameters
joblib.dump(scaler_ens, '../data/ensemble_normalization_scaler_20240227.pkl')


    # Define a function to build the model with hyperparameters
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Dense(hp.Int('units_input', min_value=3, max_value=100, step=5),
                                     activation='relu',
                                     kernel_regularizer=keras.regularizers.l1(
                                         hp.Float('l1_reg', min_value=1e-5, max_value=1e-2, sampling='log')),
                                     input_shape=(X_train_ens_norm.shape[1],)))
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(keras.layers.Dense(hp.Int(f'units_layer_{i}', min_value=3, max_value=100, step=3),
                                         activation='relu',
                                         kernel_regularizer=keras.regularizers.l1(
                                             hp.Float(f'l1_reg_layer_{i}', min_value=1e-5, max_value=1e-2,
                                                      sampling='log'))))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

        # Set up Keras Tuner


tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=30,
        executions_per_trial=1,
        directory='keras_tuner',
        project_name='secchi_depth_tuning_ensemble'
    )

    # Execute the tuning
tuner.search(X_train_ens_norm, y_train_ens, epochs=500, validation_data=(X_val_ens_norm, y_val_ens),
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)])

    # Get the best model
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
# Evaluate the best model
y_pred = best_model.predict(X_test_ens_norm)
print(f"R^2 Score on Test Set: {r2_score(y_test_ens, y_pred)}")

joblib.dump(best_model, '../models/ensemble_best_model_20240227.pkl')



