import numpy as np
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import f1_score

import os


def f1_metric(pred):
    """
    
    """
    label = pred.label_ids
    pred = pred.predictions.argmax(-1)

    f1 = f1_score(label, pred, average='macro')

    return {'f1_score': f1}


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
    
    # reading each of the subsets within their respective directories within the `../data/preprocessed/` directory
    # lendo cada um dos subsets dentro de seus respectivos diretórios dentro do diretório `../data/preprocessed/`
    train_set = Dataset.load_from_disk(os.path.join(PATH, 'train_dataset'))
    valid_set = Dataset.load_from_disk(os.path.join(PATH, 'valid_dataset'))
    test_set = Dataset.load_from_disk(os.path.join(PATH, 'test_dataset'))
        
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
        train_dataset=train_set,
        eval_dataset=valid_set,
    )
    trainer.train()
    
    # Evaluating model performance with fine-tuning in the train and validation set
    # Avaliando o desempenho do modelo com fine-tuning no train e validation set
    print(f'Train set evaluate: {trainer.evaluate(train_set)}\nValid set evaluation: {trainer.evaluate(valid_set)}')
    
    # Evaluating model performance with fine-tuning in the test set
    # Avaliando o desempenho do modelo com fine-tuning no test set
    print(f'Test set evaluate: {trainer.evaluate(test_set)}')
