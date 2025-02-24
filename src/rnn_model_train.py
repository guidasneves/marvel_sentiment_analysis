# Packages used in the system
# Pacotes utilizados no sistema
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, GlobalAveragePooling1D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.backend import clear_session
from skopt import gp_minimize

import matplotlib.pyplot as plt
import os
import sys

PROJECT_ROOT = os.path.abspath( # Getting Obtaining the absolute normalized version of the project root path (Obtendo a versão absoluta normalizada do path raíz do projeto)
    os.path.join( # Concatenating the paths (Concatenando os paths)
        os.path.dirname(__file__), # Getting the path of the scripts directory (Obtendo o path do diretório dos scripts do projeto)
        os.pardir # Gettin the constant string used by the OS to refer to the parent directory (Obtendo a string constante usada pelo OS para fazer referência ao diretório pai)
    )
)
# Adding path to the list of strings that specify the search path for modules
# Adicionando o path à lista de strings que especifica o path de pesquisa para os módulos
sys.path.append(PROJECT_ROOT)
from src.eda import *

def create_batch_dataset(dataset, batch_size=64, buffer_size=10000, shuffle=False):
    """
    [EN-US]
    Transforms a numpy array, a pandas.DataFrame, or a list into a tensorflow.data.Dataset object,
    applies pre-processing and optimizes performance.
    
    [PT-BR]
    Transforma uma matriz numpy, um pandas.DataFrame ou uma lista em um objeto tensorflow.data.Dataset,
    aplica os pré-processamentos e otimizam o desempenho.
    
    Arguments:
        dataset (numpy.array or pandas.DataFrame or list): dataset with features X and labels y for creating tensorflow.data.Dataset
                                                           (dataset que com as features X e os labels y para a criação do tensorflow.data.Dataset).
        batch_size (int, optional): size of dataset mini-batches. Defaults to 64
                                    (tamanho dos mini-batches do dataset. Padrão para 64).
        buffer_size (int, optional): elements that will initially be left out and one of them is randomly chosen as part of the random dataset. Defaults to 10000
                                     (elementos que serão inicialmente deixados de lado e um deles é escolhido aleatoriamente como parte do dataset aleatorio. Padrão para 10000).
        shuffle (boolean, optional): if True, the dataset will be shuffled, otherwise not. Defaults to False
                                     (caso seja True, o dataset será embaralhado, caso contrário, não. Padrão para False).
    
    Return:
        dataset_final (tensorflow.data.Dataset): Preprocessed tensorflow.data.Dataset
                                                 (tensorflow.data.Dataset pré-processado).
    """
    # Transforming the dataset into a tensorflow.data.Dataset object
    # Transformando o dataset em um objeto tensorflow.data.Dataset
    dataset_final = Dataset.from_tensor_slices((dataset[:, :-1], dataset[:, -1:]))
    
    # If shuffle, shuffles the dataset
    # Se shuffle, embaralha o dataset    
    if shuffle:
        dataset_final = dataset_final.shuffle(buffer_size)
    # Applying the final preprocessing
    # Aplicando os pré-processamentos finais
    dataset_final = (dataset_final
                    .batch(batch_size) # Creating batches of this dataset (Criando batches desse dataset)
                    .prefetch(AUTOTUNE) # Allowing parallel execution of this dataset (Permitindo a execução paralela dessa dataset)
                    .cache() # Storing elements in memory (Armazenando elementos na memória)
                    )
    
    return dataset_final


def create_and_compile_model(
    input_shape,
    vocab_size,
    lr=1e-3,
    embedding_dim=1000,
    dropout_rate=.1,
    lstm_1=128,
    lstm_2=64,
    dense_1=64,
    dense_2=32
):
    """
    [EN-US]
    Creates and compiles the model for training and inference.

    [PT-BR]
    Cria e compila o modelo para o treinamento e inferência.

    Arguments:
        input_shape (tuple): tuple containing the shape of the input, discarding the batch size
                             (tupla contendo o shape do input, descartando o tamanho do batch).
        vocab_size (int): size of the vocabulary used to tokenize the dataset that will be used to scale the embedding
                          (tamanho do vocabulário usado para tokenizar o dataset, ele será usado para dimensionar o embedding).
        lr (float, optional): the learning rate. Defaults to 1e-3
                              (o learning rate. Padrão para 1e-3).
        embedding_dim (int, optional): embedding vector dimension size. Defaults to 1000
                                       (tamanho da dimensão do embedding vector. Padrão para 1000).
        dropout_rate (float, optional): float between 0 and 1. Fraction of the input units to drop. Defaults to 0.1
                                        (flutuar entre 0 e 1. Fração das unidades de entrada a serem eliminadas. Padrão para 0.1).
        lstm_1 (int, optional): number of neurons for layer lstm_1. Defaults to 128
                                (quantidade de neurônios para a layer lstm_1. Padrão para 128).
        lstm_2 (int, optional): number of neurons for layer lstm_2. Defaults to 64
                                (quantidade de neurônios para a layer lstm_2. Padrão para 64).
        dense_1 (int, optional): number of neurons for layer dense_1. Defaults to 128
                                 (quantidade de neurônios para a layer dense_1. Padrão para 128).
        dense_2 (int, optional): number of neurons for layer dense_2. Defaults to 64
                                 (quantidade de neurônios para a layer dense_2. Padrão para 64).

    Return:
        model (tensorflow.keras.Model): model defined and compiled
                                        (modelo definido e compilado),
    """
    # Setting a seed
    # Definindo uma seed
    tf.random.set_seed(42)
    
    # Clearing all internal variables
    # Limpando todas as variáveis internas
    clear_session()
    
    # Setting the model architecture
    # Definindo a arquitetura do modelo    
    input_layer = Input(shape=(input_shape,))
    X = Embedding(vocab_size, embedding_dim, name='embedding_layer')(input_layer)
    #X = tf.keras.layers.Conv1D(filters=32, kernel_size=7, activation='relu')(X)
    X = Bidirectional(LSTM(lstm_1, return_sequences=True), name='bi_lstm_layer_1')(X)
    X = Dropout(rate=dropout_rate, name='dropout_layer_1')(X)
    #X = BatchNormalization(axis=-1)(X)
    X = Bidirectional(LSTM(lstm_2, return_sequences=False), name='bi_lstm_layer_2')(X)
    X = Dropout(rate=dropout_rate, name='dropout_layer_2')(X)
    #X = BatchNormalization(axis=-1)(X)
    #X = Bidirectional(LSTM(64), name='bi_lstm_layer_3')(X)
    #X = Dropout(rate=dropout_rate, name='dropout_layer_3')(X)
    #X = BatchNormalization(axis=-1)(X)
    #X = GlobalAveragePooling1D()(X)
    X = Dense(dense_1, activation='relu', name='dense_layer_1')(X)
    X = BatchNormalization()(X)
    X = Dense(dense_2, activation='relu', name='dense_layer_2')(X)
    X = BatchNormalization()(X)
    #X = Dense(128, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(lambda_r), name='dense_layer_3')(X)
    output = Dense(1, activation='linear', name='output_layer')(X)
    # Setting the input and output of the model
    # Definindo o input e o output do modelo
    model = Model(inputs=input_layer, outputs=output, name='LSTM_Bidirectional')

    # Compiling the model
    # Compilando o modelo
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def plot_history(history, metric_name, name=None):
    """
    [EN-US]
    Plots the loss and evaluation metric history for the 
    training and validation set during model training per epoch.
    
    [PT-BR]
    Plota o histórico da loss e da métrica de avaliação para o 
    training e o validation set durante o treinamento do modelo por epoch.

    Arguments:
        history (tensorflow.keras.callbacks.History): the History object gets returned by the fit() method of models
                                                      (the History object gets returned by the fit() method of models).
        metric_name (str): name of the evaluation metric used in compiling the model
                           (nome da métrica de avaliação utilizada na compilação do modelo).
        name (str, optional): name that the plot will be saved. Defaults to None
                              (nome que o gráfico será salvo. Padrão para None).
    """
    # Accessing the vector with the training loss history and validation set
    # Acessando o vetor com o histórico da loss do training e validation set
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    # Accessing the vector with the training metric history and validation set
    # Acessando o vetor com o histórico da métrica do training e validation set
    metric = history.history[metric_name]
    val_metric = history.history[f'val_{metric_name}']
    # Selecting the number of epochs
    # Selecionando a quantidade de epochs
    epochs = range(len(loss))
    # Setting lists with values to plot
    # Definindo as listas com os valores para plotar
    utils = [loss, 'loss'], [metric, metric_name]
    val_utils = [val_loss], [val_metric]

    # Defining the figure and creating the plots
    # Definindo a figura e criando os plots
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    for i in range(2):
        fig.suptitle('Performance per Epoch', fontsize=16)
        # Plotting with all epochs
        # Plotando com todas as epochs
        ax[i, 0].plot(epochs, utils[i][0], label='Train', color='cornflowerblue')
        ax[i, 0].plot(epochs, val_utils[i][0], label='Validation', color='chocolate')
        ax[i, 0].set_ylabel(utils[i][1], fontsize=16)        

        # Plotting only the final 25% of the epoch
        # PLotando apenas os 25% final da epoch
        ax[i, 1].plot(epochs, utils[i][0], label='Train', color='cornflowerblue')
        ax[i, 1].plot(epochs, val_utils[i][0], label='Validation', color='chocolate')
        ax[i, 1].set_xlim(int((len(utils[i][0]) * .75)), len(utils[i][0]))
        if i == 1:
            ax[i, 0].set_xlabel('epochs', fontsize=16)
            ax[i, 1].set_xlabel('epochs', fontsize=16)
    plt.legend(loc='best', fontsize=16)
    # If a name is passed, the plot is saved
    # Se algum nome for passado o plot é salvo
    if name:
        save_plot(name)
    plt.show()


def hyperparams_tune(hyperparams):
    """
    [EN-US]
    Setting the model for hyperparameter optimization.

    [PT-BR]
    Define o modelo para a otimização dos hiperparâmetros.

    Argument:
        hyperparams (list or numpy.array): list of hyperparameter value ranges to be optimized
                                           (lista com as faixas de valores dos hiperparâmetros para serem otimizados).

    Return:
        -metric (float): performance metric times negative
                         (métrica de avaliação vezes negativo). 
    """
    # Setting the hyperparameters
    # Definindo os hiperparâmetros
    lr = hyperparams[0]
    embedding_dim = hyperparams[1]
    dropout_rate = hyperparams[2]
    lstm_1 = hyperparams[3]
    lstm_2 = hyperparams[4]
    dense_1 = hyperparams[5]
    dense_2 = hyperparams[6]
    
    # Setting the MAX_LEN
    # Definindo o MAX_LEN
    train_batch = next(X_train_opt.as_numpy_iterator())
    MAX_LEN = train_batch[0].shape[1]
    # Loading vocabulary from trained tokenizer
    # Carregando o vocabulário do tokenizer treinado
    VOCAB_SIZE = len(pickle.load(open(os.path.join(PATH_M, 'vectorizer.pkl'), 'rb'))['vocabulary'])
    
    # Defining the model to perform the optimization
    # Definindo o modelo para performar a otimização
    model = create_and_compile_model(
        MAX_LEN,
        VOCAB_SIZE,
        lr=lr,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        lstm_1=lstm_1,
        lstm_2=lstm_2,
        dense_1=dense_1,
        dense_2=dense_2
    )
    model.fit(X_train_opt, validation_data=X_valid_opt, epochs=25, verbose=0)

    # Computing performance on the validation set
    # Calculando o desempenho no validation set
    metric = model.evaluate(valid_set, verbose=0)[1]

    # -metric, because we want the pair of hyperparameters that minimizes the metric
    # -metric, porque queremos o par de hiperparâmetros que minimize a métrica
    return -metric


if __name__ == '__main__':
    # Setting the global variables `PATH` and `PATH_M`,
    # with the path of the directory where the data will be loaded and the path of the model weights
    # Configurando as variáveis globais `PATH` e `PATH_M`,
    # com o path do diretório onde os dados serão carregados e o path dos pesos modelos
    PATH = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'data',
            'preprocessed'
        )
    )
    PATH_M = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            'models'
        )
    )
    
    # Subset names
    # Nomes dos subsets
    files = ['train', 'valid', 'test']
    datasets = []
    # Looping through each name
    # Percorrendo cada nome
    for file in files:
        # Reading each subset and adding it to the list for later extraction
        # Lendo cada subset e adicionando na lista para extração posteriormente
        with open(os.path.join(PATH, f'{file}_tokens.npy'), 'rb') as f:
            datasets.append(np.load(f))
    # Extracting each subset from the `datasets` list
    # Extraindo cada subset da lista `datasets`
    train_corpus, valid_corpus, test_corpus = datasets

    # Dataset global variables
    # Variáveis globais do dataset
    BATCH_SIZE = 128
    BUFFER_SIZE = 1000
    
    # Model global variables
    # Variáveis globais do modelo
    MAX_LEN = train_corpus.shape[1] - 1
    EMBEDDING_DIM = 5000
    DROPOUT_RATE = .1
    # Loading vocabulary from trained tokenizer
    # Carregando o vocabulário do tokenizer treinado
    VOCAB_SIZE = len(pickle.load(open(os.path.join(PATH_M, 'vectorizer.pkl'), 'rb'))['vocabulary'])

    # Creating `tensorflow.data.Dataset` for each subset
    # Criando o `tensorflow.data.Dataset` para cada subset
    train_set = create_batch_dataset(train_corpus, BATCH_SIZE, BUFFER_SIZE, shuffle=True)
    valid_set = create_batch_dataset(valid_corpus, BATCH_SIZE, BUFFER_SIZE)
    test_set = create_batch_dataset(test_corpus, BATCH_SIZE, BUFFER_SIZE)

    # Defining the range for testing each hyperparameter
    # Definindo a faixa para teste de cada hiperparâmetro
    space = [
        (1e-6, 1e-1, 'log-uniform'), # learning rate
        (100, 5000), # Embedding dimension
        (.2, .8), # Dropout rate
        (16, 128), # First LSTM units
        (16, 128), # Second LSTM units
        (16, 128), # First Dense units
        (16, 128), # Second Dense units
    ]
    
    # Copying the training and validation subset to optimize the hyperparameters
    # Copiando o subset de treino e de validação para otimizar os hiperparâmetros
    X_train_opt = tf.identity(train_set)
    X_valid_opt = tf.identity(valid_set)
    # Performing Bayesian optimization
    # Performando a bayesian optimization
    opt = gp_minimize(
        hyperparams_tune,
        space,
        random_state=42,
        verbose=0,
        n_calls=10,
        n_random_starts=4
    )

    # Best hyperparameters
    # Melhores hiperparâmetros
    lr, emb, drp, lstm_1, lstm_2, fc_1, fc_2 = opt.x
    # Setting the model
    # Definindo o modelo
    model = create_and_compile_model(
        MAX_LEN,
        VOCAB_SIZE,
        lr=lr,
        embedding_dim=emb,
        dropout_rate=drp,
        lstm_1=lstm_1,
        lstm_2=lstm_2,
        dense_1=dense_1,
        dense_2=dense_2,
    )
    # Plotting the model summary
    # Plotando o resumo modelo
    print(model.summary())
        
    # Setting the callbacks
    # Definindo os callbacks
    checkpoint_cb = ModelCheckpoint(os.path.join(PATH_M, 'lstm_model.keras'), save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=50, restore_best_weights=True)
    
    # Training the model
    # Treinando o modelo
    print('Training the model...')
    history = model.fit(
        train_set,
        epochs=25,
        validation_data=valid_set,
        callbacks=[checkpoint_cb, early_stopping_cb],
        verbose=2
    )

    # Plotting the training models history
    # PLotando o histórico de treinamento do modelo
    plot_history(history, 'accuracy')

    # Evaluating the model on training and validation data
    # Avaliando o modelo nos dados de treino e de validação
    print(f'Train set evaluate: {model.evaluate(train_set, verbose=0)[1]}')
    print(f'Validation set evaluate: {model.evaluate(valid_set, verbose=0)[1]}')
