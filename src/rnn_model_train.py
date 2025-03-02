# Packages used in the system
# Pacotes utilizados no sistema
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import clear_session
from skopt import gp_minimize
from sklearn.decomposition import KernelPCA

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
    embedding_dim=32,
    dropout_rate=[.5, .4]
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
        embedding_dim (int, optional): embedding vector dimension size. Defaults to 32
                                       (tamanho da dimensão do embedding vector. Padrão para 32).
        dropout_rate (array, optional): vector with two elements, one for each layer.. Fraction of the input units to drop. Defaults to [0.5 and 0.4]
                                        (vetor com dois elementos, uma para cada layer. Fração das unidades de entrada a serem eliminadas. Padrão para [0.5 e 0.4]).

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
    X = Dropout(rate=dropout_rate[0])(X)
    X = BatchNormalization()(X)
    X = Bidirectional(LSTM(2, return_sequences=False), name='lstm_bidirectional_layer')(X)
    X = Dropout(rate=dropout_rate[1])(X)
    X = BatchNormalization()(X)
    output = Dense(1, activation='sigmoid', name='output_layer')(X)
    # Setting the input and output of the model
    # Definindo o input e o output do modelo
    model = Model(inputs=input_layer, outputs=output)

    # Compiling the model
    # Compilando o modelo
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=BinaryCrossentropy(),
        metrics=[F1Score(average='macro', threshold=.5)]
    )

    return model


def hyperparams_tune(hyperparams):
    """
    [EN-US]
    Sets the model for hyperparameter optimization.

    [PT-BR]
    Define o modelo para a otimização dos hiperparâmetros.

    Argument:
        hyperparams (list or numpy.array): list of hyperparameter value ranges to be optimized
                                           (lista com as faixas de valores dos hiperparâmetros para serem otimizados).

    Return:
        -metric (float): performance metric times negative
                         (métrica de avaliação vezes negativo). 
    """
    # Setting the hyperparameter
    # Definindo o hiperparâmetro
    lr = hyperparams[0]
    
    # Defining the model to perform the optimization
    # Definindo o modelo para performar a otimização
    model = create_and_compile_model(
        MAX_LEN,
        VOCAB_SIZE,
        lr=lr,
    )
    model.fit(train_set, validation_data=val_set, epochs=15, verbose=0)

    # Computing performance on the validation set
    # Calculando o desempenho no validation set
    metric = model.evaluate(val_set, verbose=0)[1]

    # -metric, because we want the pair of hyperparameters that minimizes the metric
    # -metric, porque queremos o par de hiperparâmetros que minimize a métrica
    return -metric


def plot_history(history, metric_name, colors=['b', 'r'], name=None):
    """
    [EN-US]
    Plots the loss and evaluation metric history for the training and validation set,
    during model training per epoch.
    
    [PT-BR]
    Plota o histórico da loss e da métrica de avaliação para o training e o validation set,
    durante o treinamento do modelo por epoch.

    Arguments:
        history (tensorflow.keras.callbacks.History): the History object gets returned by the fit() method of models
                                                      (the History object gets returned by the fit() method of models).
        metric_name (str): name of the evaluation metric used in compiling the model
                           (nome da métrica de avaliação utilizada na compilação do modelo).
        colors (list, optional): list with 2 colors of arguments for the plot. Defaults to ['b', 'r']
                                 (lista com 2 cores de argumentos para o plot. Padrão para ['b', 'r']).
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

    # Setting the plot style
    # Definindo o estilo do plot
    plt.style.use('default')
    # Defining the figure and creating the plots
    # Definindo a figura e criando os plots
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    for i in range(2):
        fig.suptitle('Performance per Epoch', fontsize=16)
        # Plotting with all epochs
        # Plotando com todas as epochs
        ax[i, 0].plot(epochs, utils[i][0], label='Train', color=colors[0], linewidth=2)
        ax[i, 0].plot(epochs, val_utils[i][0], label='Validation', color=colors[1], linewidth=2)
        ax[i, 0].set_ylabel(utils[i][1], fontsize=16)        

        # Plotting only the final 25% of the epoch
        # PLotando apenas os 25% final da epoch
        ax[i, 1].plot(epochs, utils[i][0], label='Train', color=colors[0], linewidth=2)
        ax[i, 1].plot(epochs, val_utils[i][0], label='Validation', color=colors[1], linewidth=2)
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


def plot_word_embeddings(
    embeddings,
    pos_corpus,
    neg_corpus,
    vocabulary,
    idx=0,
    words_to_plot=15,
    name=None,
    colors=['b', 'r']
):
    """
    [EN-US]
    Returns a scatterplot with the word embeddings in a 2D space.
    
    [PT-BR]
    Retorna um gráfico de dispersão com as word embeddings em um espaço 2D.
    
    Arguments:
        embeddings (numpy.array): the weights of the embedding layer
                                  (os pesos da embedding layer).
        pos_corpus (numpy.array): array with positive tokenized examples
                                  (array com os exemplos tokenizados positivos).
        neg_corpus (numpy.array): array with negative tokenized examples
                                  (array com os exemplos tokenizados negativos).
        vocabulary (array): vocabulary trained on the training set
                            (vocabulário treinado sobre o training set).
        idx (int, optional): example of the corpus that the embeddings will be extracted to plot. Defaults to 0
                             (exemplo do corpus que as embeddings serão extraídas para plotar. Padrão para 0).
        words_to_plot (int, optional): number of tokens to extract from each corpus. Defaults to 15
                                       (quantidade de tokens para extrair de cada corpus. Padrão para 15).
        name (str, optional): name that the plot will be saved. Defaults to None
                              (nome que o gráfico será salvo. Padrão para None).
        colors (list, optional): list with 2 colors of arguments for the plot. Defaults to ['b', 'r']
                                 (lista com 2 cores de argumentos para o plot. Padrão para ['b', 'r']).
    """
    # Selecting the example `idx` through the token `words_to_plot` from each corpus to plot their respective embeddings
    # Selecionando o exemplo `idx` até o token `words_to_plot` de cada corpus para plotar seus respectivos embeddings
    pos_seq = pos_corpus[idx][:words_to_plot]
    neg_seq = neg_corpus[idx][:words_to_plot]

    # Computing the dimensionality reduction of embeddings
    # Computando a redução de dimensionalidade dos embeddings
    pca = KernelPCA(
        n_components=2,
        kernel='rbf',
        gamma=.9,
        n_jobs=-1,
        random_state=42
    )
    embeddings_2D = pca.fit_transform(embeddings)

    # Setting the plot style
    # Definindo o estilo do plot
    plt.style.use('fivethirtyeight')
    # Plotting the embeddings
    # Plotando os embeddings
    plt.figure(figsize=(8, 8))
    # Scatter plot for positive words
    # Gráfico de dispersão para palavras positivas
    plt.scatter(embeddings_2D[pos_seq][:, 0], embeddings_2D[pos_seq][:, 1], color=colors[0], label='Action')
    for i, token in enumerate(pos_seq):
        plt.annotate(vocabulary[token], (embeddings_2D[pos_seq][i, 0], embeddings_2D[pos_seq][i, 1]))

    # Scatter plot for negative words
    # Gráfico de dispersão para palavras negativas
    plt.scatter(embeddings_2D[neg_seq][:, 0], embeddings_2D[neg_seq][:, 1], color=colors[1], label='Non-action')
    for i, token in enumerate(neg_seq):
        plt.annotate(vocabulary[token], (embeddings_2D[neg_seq][i, 0], embeddings_2D[neg_seq][i, 1]))
    plt.title('Word Embeddings in 2D', fontsize=16)
    plt.legend(loc='best', fontsize=16, title='Labels')
    # If a name is passed, the plot is saved
    # Se algum nome for passado o plot é salvo
    if name:
        save_plot(name)
    plt.show()


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
    files = ['train', 'validation', 'test']
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
    train_corpus, val_corpus, test_corpus = datasets

    # Dataset global variables
    # Variáveis globais do dataset
    BATCH_SIZE = 32
    BUFFER_SIZE = 1000
    
    # Model global variables
    # Variáveis globais do modelo
    tokenizer = pickle.load(open(os.path.join(PATH_M, 'vectorizer.pkl'), 'rb'))
    MAX_LEN = tokenizer['config']['output_sequence_length']
    EMBEDDING_DIM = 16
    DROPOUT_RATE = [.5, .5]
    LR = 5e-3
    VOCABULARY = tokenizer['vocabulary']
    VOCAB_SIZE = len(VOCABULARY)

    # Creating `tensorflow.data.Dataset` for each subset
    # Criando o `tensorflow.data.Dataset` para cada subset
    train_set = create_batch_dataset(train_corpus, BATCH_SIZE, BUFFER_SIZE, shuffle=True)
    val_set = create_batch_dataset(val_corpus, BATCH_SIZE, BUFFER_SIZE, shuffle=True)
    test_set = create_batch_dataset(test_corpus, BATCH_SIZE, BUFFER_SIZE, shuffle=True)

    # Defining the range for testing each hyperparameter
    # Definindo a faixa para teste de cada hiperparâmetro
    space = [
        (1e-4, 1e-1, 'log-uniform') # learning rate
    ]
    # Performing Bayesian optimization
    # Performando a bayesian optimization
    opt = gp_minimize(
        hyperparams_tune,
        space,
        random_state=42,
        verbose=0,
        n_calls=5,
        n_random_starts=2
    )
    # Best hyperparameter
    # Melhor hiperparâmetro
    #LR = opt.x
    
    # Setting the model
    # Definindo o modelo
    model = create_and_compile_model(
        MAX_LEN,
        VOCAB_SIZE,
        lr=LR,
        embedding_dim=EMBEDDING_DIM,
        dropout_rate=DROPOUT_RATE
    )
    # Plotting the model summary
    # Plotando o resumo modelo
    print(model.summary())
        
    # Setting the callbacks
    # Definindo os callbacks
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=.5, patience=2)
    checkpoint_cb = ModelCheckpoint(os.path.join(PATH_M, 'lstm_model.keras'), save_best_only=True)
    early_stopping_cb = EarlyStopping(patience=5, restore_best_weights=True)
    
    # Training the model
    # Treinando o modelo
    print('Training the model...')
    history = model.fit(
        train_set,
        epochs=10,
        verbose=2,
        validation_data=val_set,
        callbacks=[reduce_lr_cb, checkpoint_cb, early_stopping_cb]
    )

    # Plotting the training models history
    # PLotando o histórico de treinamento do modelo
    plot_history(history, 'f1_score', name='model_history')

    # Evaluating the model on training and validation data
    # Avaliando o modelo nos dados de treino e de validação
    print(f'Train set evaluate: {model.evaluate(train_set, verbose=0)[1]:.4f}')
    print(f'Validation set evaluate: {model.evaluate(val_set, verbose=0)[1]:.4f}')
    # Evaluating the final model on the test set
    # Avaliando o modelo final no test set 
    print(f'Test set evaluate: {model.evaluate(test_set, verbose=0)[1]:.4f}')

    # Concatenating all datasets to plot word embeddings
    # Concatenando todos os datasets para plotar as word embeddings
    dataset = np.concatenate(datasets)
    print(f'Dataset dimension: {dataset.shape}')

    # Selecting the corpus with only the positive and negative class
    # Selecionando o corpus apenas com a classe positiva e negativa
    pos_corpus = dataset[dataset[:, -1] == 1][:, :-1]
    neg_corpus = dataset[dataset[:, -1] == 0][:, :-1]

    # Obtaining the embedding layer and its weights to plot the embedding of each corresponding word
    # Obtendo a embedding layer e seus pesos para plotar o embedding de cada palavra correspondente
    embedding_layer = model.get_layer('embedding_layer')
    embeddings = embedding_layer.get_weights()[0]

    # Setting the global variable with the colors for the word embeddings plot
    # Definindo a variável global com as cores para o plot das word embeddings
    COLORS = ['cornflowerblue', 'chocolate']
    # Plotting the word embeddings of the positive corpus and the negative corpus
    # Plotando as word embeddings do corpus positivo e do corpus negativo
    plot_word_embeddings(
        embeddings,
        pos_corpus,
        neg_corpus,
        VOCABULARY,
        idx=3,
        words_to_plot=10,
        name='word_embeddings_2d',
        colors=COLORS
    )
