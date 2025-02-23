import numpy as np
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments

import os

# Setting the global variables `PATH` and `PATH_M`,
# with the path of the directory where the data will be loaded and the path of the model weights
# Configurando as variáveis globais `PATH` e `PATH_M`,
# com o path do diretório onde os dados serão carregados e o path dos pesos modelos
PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
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
# Momes dos subsets
files = ['train', 'valid', 'test']
datasets = []
# Looping through each name
# Percorrendo cada nome
for file in files:
    # Reading each subset and adding it to the list for later extraction
    # Lendo cada subset e adicionando na lista para extração posteriormente
    with open(os.path.join(PATH, f'{file}_transformers.npy'), 'rb') as f:
        datasets.append(np.load(f))
# Extracting each subset from the `datasets` list
# Extraindo cada subset da lista `datasets`
train_corpus, valid_corpus, test_corpus = datasets

# Defining the pre-trained model that we will do the fine-tuning
# Definindo o modelo pré-treinado que faremos o fine-tuning
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
# Fine-tuning hyperparameters
# Hiperparâmetros do fine-tuning
training_args = TrainingArguments(
    output_dir=os.path.join(PATH_M, 'transformers_results'),
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=20,
    weight_decay=0.01,
    logging_steps=50
)

# Trainer object
# Objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_corpus,
    eval_dataset=valid_corpus,
)
trainer.train()

# Evaluating model performance with fine-tuning in the test set
# Avaliando o desempenho do modelo com fine-tuning no test set
print(f'Test set evaluate: {trainer.evaluate(test_corpus)}')
