import pandas as pd
import numpy as np
from tensorflow.data import Dataset, AUTOTUNE
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt


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


#if __name__ == '__main__':
    