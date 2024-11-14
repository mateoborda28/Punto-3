import pandas as pd
import numpy as np
import seaborn as sns

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Bidirectional, GRU, Input,Concatenate
from tensorflow.keras.layers import Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import silhouette_score
from collections import Counter

from sklearn.metrics import accuracy_score, roc_auc_score
path = "C:/Users/mbr19/OneDrive/Escritorio/Ultimo semestre/Python/Tercer Corte/Trabajos/Punto 3/"

def evaluacion():

    embedding_vector_length_bert = 768
    embedding_vector_length_fast = 300

###### preparación para Bert ###### 
    embedding_bert_entrenamiento = pd.read_csv(path + 'embedding_train_df.csv')
    embedding_bert_prueba = pd.read_csv(path + 'embedding_test_df.csv')
###### preparación para Fast ######
    embedding_fast_entrenamiento = pd.read_csv(path + 'embedding_fast_train.csv')
    embedding_fast_prueba = pd.read_csv(path + 'embedding_fast_test.csv')
###### preparación para ingenieria ######
    ingenieria_entrenamiento = pd.read_csv("ingenieria_de_variables_train.csv")
    ingenieria_prueba = pd.read_csv("ingenieria_de_variables_test.csv")

###### preparación sin ingenieria ######
    entrenamiento_sin_ingenieria = pd.read_csv(path + 'entrenamiento_sin_ingenieria.csv')
    prueba_sin_ingenieria = pd.read_csv(path + 'prueba_sin_ingenieria.csv')


    ### Sin ingenieria
    ## Train
    features_sin_ingenieria = entrenamiento_sin_ingenieria.columns[entrenamiento_sin_ingenieria.dtypes != 'object'].tolist()
    features_train = entrenamiento_sin_ingenieria.get(features_sin_ingenieria)
    y_train = entrenamiento_sin_ingenieria.get(["Sarcasmo"]).to_numpy()
    
    ## Test
    features_test_sin_ingenieria = prueba_sin_ingenieria.columns[prueba_sin_ingenieria.dtypes != 'object'].tolist()
    features_test = prueba_sin_ingenieria.get(features_test_sin_ingenieria)
    y_test = prueba_sin_ingenieria.get(["Sarcasmo"]).to_numpy()
    
  
    ### Con ingenieria
    ## Train
    features_con_ingenieria = ingenieria_entrenamiento.columns[ingenieria_entrenamiento.dtypes != 'object'].tolist()
    features_train_ingenieria = ingenieria_entrenamiento.get(features_con_ingenieria)
    y_train_ingenieria = ingenieria_entrenamiento.get(["Sarcasmo"]).to_numpy()

    ## Test
    features_con_ingenieria = ingenieria_prueba.columns[ingenieria_prueba.dtypes != 'object'].tolist()
    features_test_ingenieria = prueba_sin_ingenieria.get(features_con_ingenieria)
    y_test_ingenieria = ingenieria_prueba.get(["Sarcasmo"]).to_numpy()

######################
    features_train.columns = features_train.columns.astype(str)
    features_test.columns = features_test.columns.astype(str)
    features_train_ingenieria.columns = features_train_ingenieria.columns.astype(str)
    features_test_ingenieria.columns = features_test_ingenieria.columns.astype(str)
#####################

    scaler = MinMaxScaler((-1.0,1.0))
    features_train_scaled = pd.DataFrame(scaler.fit_transform(features_train))
    features_train_scaled.columns = features_train.columns
    features_test_scaled = pd.DataFrame(scaler.fit_transform(features_test))
    features_test_scaled.columns = features_test.columns
    features_train_ingenieria_scaled = pd.DataFrame(scaler.fit_transform(features_train_ingenieria))
    features_train_ingenieria_scaled.columns = features_train_ingenieria.columns
    features_test_ingenieria_scaled = pd.DataFrame(scaler.fit_transform(features_test_ingenieria))
    features_test_ingenieria_scaled.columns = features_test_ingenieria.columns


##### red neuronal Bert #####
### Sin ingenieria
    x1_bert = Input(shape=(embedding_vector_length_bert,), name='Input_Embedding')
    x2_bert = Input(shape=(features_train_scaled.shape[1],), name='Input_Features')
    # Capa entrada
    x_bert = Concatenate(name='Concatenar')([x1_bert, x2_bert])
    x_bert = Dropout(0.50)(x_bert)
    # capas ocultas
    x_bert = Dense(64, activation='elu', name='Capa_Densa_1')(x_bert)
    x_bert = Dropout(0.25)(x_bert)
    x_bert = Dense(32, activation='elu', name='Capa_Densa_2')(x_bert)
    x_bert = Dropout(0.25)(x_bert)
    x_bert = Dense(16, activation='elu', name='Capa_Densa_3')(x_bert)
    x_bert = Dropout(0.25)(x_bert)
    # Capa de salida para clasificación binaria
    x_bert = Dense(1, activation='sigmoid', name='Output')(x_bert)
    model1 = Model(inputs=[x1_bert, x2_bert], outputs=x_bert)
    model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model1.fit(
    x=[embedding_bert_entrenamiento, features_train_scaled],  # Entrenamos con embeddings y features escaladas
    y=y_train,
    validation_data=([embedding_bert_prueba, features_test_scaled], y_test),  # Validación con datos de prueba
    epochs=100,
    batch_size=32,
    verbose=1)


### Con ingenieria
    x3_bert = Input(shape=(embedding_vector_length_bert,), name='Input_Embedding')
    x4_bert = Input(shape=(features_train_ingenieria_scaled.shape[1],), name='Input_Features')
    # Capa entrada
    x1_bert = Concatenate(name='Concatenar')([x3_bert, x4_bert])
    x1_bert = Dropout(0.50)(x1_bert)
    # capas ocultas
    x1_bert = Dense(64, activation='elu', name='Capa_Densa_1')(x1_bert)
    x1_bert = Dropout(0.25)(x1_bert)
    x1_bert = Dense(32, activation='elu', name='Capa_Densa_2')(x1_bert)
    x1_bert = Dropout(0.25)(x1_bert)
    x1_bert = Dense(16, activation='elu', name='Capa_Densa_3')(x1_bert)
    x1_bert = Dropout(0.25)(x1_bert)
    # Capa de salida para clasificación binaria
    x1_bert = Dense(1, activation='sigmoid', name='Output')(x1_bert)
    model2= Model(inputs=[x3_bert, x4_bert], outputs=x1_bert)
    model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model2.fit(
    x=[embedding_bert_entrenamiento, features_train_ingenieria_scaled],  # Entrenamos con embeddings y features escaladas
    y=y_train,
    validation_data=([embedding_bert_prueba, features_test_ingenieria_scaled], y_test),  # Validación con datos de prueba
    epochs=100,
    batch_size=32,
    verbose=1)

##### red neuronal Fast #####
### Sin ingenieria
    x1_fast = Input(shape=(embedding_vector_length_fast,), name='Input_Embedding')
    x2_fast = Input(shape=(features_train_scaled.shape[1],), name='Input_Features')
    # Capa entrada
    x_fast = Concatenate(name='Concatenar')([x1_fast, x2_fast])
    x_fast = Dropout(0.50)(x_fast)
    # capas ocultas
    x_fast = Dense(64, activation='elu', name='Capa_Densa_1')(x_fast)
    x_fast = Dropout(0.25)(x_fast)
    x_fast = Dense(32, activation='elu', name='Capa_Densa_2')(x_fast)
    x_fast = Dropout(0.25)(x_fast)
    x_fast = Dense(16, activation='elu', name='Capa_Densa_3')(x_fast)
    x_fast = Dropout(0.25)(x_fast)
    # Capa de salida para clasificación binaria
    x_fast = Dense(1, activation='sigmoid', name='Output')(x_fast)
    model3 = Model(inputs=[x1_fast, x2_fast], outputs=x_fast)
    model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model3.fit(
    x=[embedding_fast_entrenamiento, features_train_scaled],  # Entrenamos con embeddings y features escaladas
    y=y_train,
    validation_data=([embedding_fast_prueba, features_test_scaled], y_test),  # Validación con datos de prueba
    epochs=100,
    batch_size=32,
    verbose=1)

### Con ingenieria
    x3_fast = Input(shape=(embedding_vector_length_fast,), name='Input_Embedding')
    x4_fast = Input(shape=(features_train_ingenieria_scaled.shape[1],), name='Input_Features')
    # Capa entrada
    x1_fast = Concatenate(name='Concatenar')([x3_fast, x4_fast])
    x1_fast = Dropout(0.50)(x1_fast)
    # capas ocultas
    x1_fast = Dense(64, activation='elu', name='Capa_Densa_1')(x1_fast)
    x1_fast = Dropout(0.25)(x1_fast)
    x1_fast = Dense(32, activation='elu', name='Capa_Densa_2')(x1_fast)
    x1_fast = Dropout(0.25)(x1_fast)
    x1_fast = Dense(16, activation='elu', name='Capa_Densa_3')(x1_fast)
    x1_fast = Dropout(0.25)(x1_fast)
    # Capa de salida para clasificación binaria
    x1_fast = Dense(1, activation='sigmoid', name='Output')(x1_fast)
    model4 = Model(inputs=[x3_fast, x4_fast], outputs=x1_fast)
    model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model4.fit(
    x=[embedding_fast_entrenamiento, features_train_ingenieria_scaled],  # Entrenamos con embeddings y features escaladas
    y=y_train,
    validation_data=([embedding_fast_prueba, features_test_ingenieria_scaled], y_test),  # Validación con datos de prueba
    epochs=100,
    batch_size=32,
    verbose=1)


# Predicciones del train para el modelo 1
    y_pred1 = model1.predict([embedding_bert_entrenamiento, features_train_scaled])
    y_true1 = y_train
    y_pred1 = (y_pred1 >= 0.5).astype(int)
    np.sum(y_pred1)
# Evaluación del train para el modelo 1
    accuracy_score1 = accuracy_score(y_true1, y_pred1)
    roc_auc_score1 = roc_auc_score(y_true1, y_pred1)

# Predicciones del train para el modelo 2
    y_pred2 = model2.predict([embedding_bert_entrenamiento, features_train_ingenieria_scaled])
    y_true2 = y_train
    y_pred2 = (y_pred2 >= 0.5).astype(int)
    np.sum(y_pred2)
# Evaluación del train para el modelo 2
    accuracy_score2 = accuracy_score(y_true2, y_pred2)
    roc_auc_score2 = roc_auc_score(y_true2, y_pred2)

# Predicciones del train para el modelo 3
    y_pred3 = model3.predict([embedding_fast_entrenamiento, features_train_scaled])
    y_true3 = y_train
    y_pred3 = (y_pred3 >= 0.5).astype(int)
    np.sum(y_pred3)
# Evaluación del train para el modelo 3
    accuracy_score3 = accuracy_score(y_true3, y_pred3)
    roc_auc_score3 = roc_auc_score(y_true3, y_pred3)

# Predicciones del train para el modelo 4
    y_pred4 = model4.predict([embedding_fast_entrenamiento, features_train_ingenieria_scaled])
    y_true4 = y_train
    y_pred4 = (y_pred4 >= 0.5).astype(int)
    np.sum(y_pred4)
# Evaluación del train para el modelo 4
    accuracy_score4 = accuracy_score(y_true4, y_pred4)
    roc_auc_score4 = roc_auc_score(y_true4, y_pred4)

# Predicciones del test para el modelo 1
    y_pred1_test = model1.predict([embedding_bert_prueba, features_test_scaled])
    y_true1_test = y_test
    y_pred1_test = (y_pred1_test >= 0.5).astype(int)
    accuracy1_test = accuracy_score(y_true1_test, y_pred1_test)
    roc1_test = roc_auc_score(y_true1_test, y_pred1_test)

# Predicciones del test para el modelo 2
    y_pred2_test = model2.predict([embedding_bert_prueba, features_test_ingenieria_scaled])
    y_true2_test = y_test
    y_pred2_test = (y_pred2_test >= 0.5).astype(int)
    accuracy2_test = accuracy_score(y_true2_test, y_pred2_test)
    roc2_test = roc_auc_score(y_true2_test, y_pred2_test)

# Predicciones del test para el modelo 3
    y_pred3_test = model3.predict([embedding_fast_prueba, features_test_scaled])
    y_true3_test = y_test
    y_pred3_test = (y_pred3_test >= 0.5).astype(int)
    accuracy3_test = accuracy_score(y_true3_test, y_pred3_test)
    roc3_test = roc_auc_score(y_true3_test, y_pred3_test)

# Predicciones del test para el modelo 4
    y_pred4_test = model4.predict([embedding_fast_prueba, features_test_ingenieria_scaled])
    y_true4_test = y_test
    y_pred4_test = (y_pred4_test >= 0.5).astype(int)
    accuracy4_test = accuracy_score(y_true4_test, y_pred4_test)
    roc4_test = roc_auc_score(y_true4_test, y_pred4_test)


# Imprimir resultados del modelo 1

    print(f"Modelo Bert sin ingenieria - Accuracy en Entrenamiento: {accuracy_score1:.4f}")
    print(f"Modelo Bert sin ingenieria - ROC AUC en Entrenamiento: {roc_auc_score1:.4f}")
    print(f"Modelo Bert sin ingenieria - Accuracy en Test: {accuracy1_test:.4f}")
    print(f"Modelo Bert sin ingenieria - ROC AUC en Test: {roc1_test:.4f}")

# Imprimir resultados del modelo 2
    print(f"Modelo Bert con ingenieria - Accuracy en Entrenamiento: {accuracy_score2:.4f}")
    print(f"Modelo Bert con ingenieria - ROC AUC en Entrenamiento: {roc_auc_score2:.4f}")
    print(f"Modelo Bert con ingenieria - Accuracy en Test: {accuracy2_test:.4f}")
    print(f"Modelo Bert con ingenieria - ROC AUC en Test: {roc2_test:.4f}")

# Imprimir resultados del modelo 3
    print(f"Modelo Fast sin ingenieria - Accuracy en Entrenamiento: {accuracy_score3:.4f}")
    print(f"Modelo Fast sin ingenieria - ROC AUC en Entrenamiento: {roc_auc_score3:.4f}")
    print(f"Modelo Fast sin ingenieria - Accuracy en Test: {accuracy3_test:.4f}")
    print(f"Modelo Fast sin ingenieria - ROC AUC en Test: {roc3_test:.4f}")

# Imprimir resultados del modelo 4
    print(f"Modelo Fast con ingenieria - Accuracy en Entrenamiento: {accuracy_score4:.4f}")
    print(f"Modelo Fast con ingenieria - ROC AUC en Entrenamiento: {roc_auc_score4:.4f}")
    print(f"Modelo Fast con ingenieria - Accuracy en Test: {accuracy4_test:.4f}")
    print(f"Modelo Fast con ingenieria - ROC AUC en Test: {roc4_test:.4f}")

# Crear un diccionario con los datos
    data = {
     'Modelo': ['BERT sin ingeniería', 'BERT con ingeniería', 'FastText sin ingeniería', 'FastText con ingeniería'],
     'Accuracy Entrenamiento': [accuracy_score1, accuracy_score2, accuracy_score3, accuracy_score4],
     'ROC AUC Entrenamiento': [roc_auc_score1, roc_auc_score2, roc_auc_score3, roc_auc_score4],
     'Accuracy Test': [accuracy1_test, accuracy2_test, accuracy3_test, accuracy4_test],
     'ROC AUC Test': [roc1_test, roc2_test, roc3_test, roc4_test]}
    resultados_df = pd.DataFrame(data)  
    resultados_df.to_csv(path + 'resultados_modelos.csv', index=False)


    return accuracy_score1, roc_auc_score1, accuracy_score2, roc_auc_score2, accuracy_score3, roc_auc_score3, accuracy_score4, roc_auc_score4, accuracy1_test, roc1_test, accuracy2_test, roc2_test, accuracy3_test, roc3_test, accuracy4_test, roc4_test, resultados_df     