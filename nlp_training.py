import spacy
from spacy.training.example import Example
import random
from nlp_training_data import treinamento_data
from spacy.training.example import Example

# Função de treinamento
def treinar_ner(nlp, treinamento, n_iter=1000):
    random.shuffle(treinamento_data)

    # Dividir em 80% treino e 20% validação
    treino_data = treinamento[:int(len(treinamento) * 0.8)]
    validacao_data = treinamento[int(len(treinamento) * 0.8):]

    optimizer = nlp.begin_training()

    # Treinamento
    for i in range(n_iter):
        # print(f"Iteração {i + 1}")
        random.shuffle(treino_data)
        losses = {}
        
        for texto, entidades in treino_data:
            doc = nlp.make_doc(texto)
            try:
                exemplo = Example.from_dict(doc, {"entities": entidades})
                nlp.update([exemplo], losses=losses)
            except Exception as e:
                print(f"Erro ao processar: texto={texto}, entidades={entidades}. Erro: {e}")
        
        if i % 10 == 0:
            validacao_performance = avaliar_modelo(nlp, validacao_data)
            print(f"Desempenho na validação após {i+1} iterações: {validacao_performance}")
        
        # print(f"Perda de treinamento: {losses}")
    
    return nlp


# Função para avaliar o modelo nos dados de validação
def avaliar_modelo(nlp, validacao_data):
    """
    Avalia o desempenho do modelo nos dados de validação.
    Retorna métricas como precisão, recall e F1.
    """
    true_entities = []
    pred_entities = []
    
    for texto, entidades in validacao_data:
        # Processar o texto com o modelo treinado
        doc = nlp(texto)
        
        # Adicionar as entidades previstas
        pred_entities.extend([(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents])
        
        # Adicionar as entidades verdadeiras
        true_entities.extend(entidades)

    # Transformar em conjuntos para evitar duplicatas
    true_entities_set = set(true_entities)
    pred_entities_set = set(pred_entities)

    # Calcular métricas
    tp = len(true_entities_set & pred_entities_set)  # Interseção: verdadeiras e previstas corretamente
    fp = len(pred_entities_set - true_entities_set)  # Previstas incorretamente
    fn = len(true_entities_set - pred_entities_set)  # Verdadeiras não previstas

    # Prevenir divisão por zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}

# Carregar o modelo de linguagem em português
nlp = spacy.load('pt_core_news_sm')

# Criar o pipeline NER se não existir
ner = nlp.create_pipe('ner') if 'ner' not in nlp.pipe_names else nlp.get_pipe('ner')

# Adicionar a nova etiqueta ao pipeline de NER
ner.add_label('NUM_PESSOAS')
ner.add_label('HORARIO')
ner.add_label('DATA')

# Treinar o modelo
modelo_treinado = treinar_ner(nlp, treinamento_data, n_iter=50)

# Salvar o modelo treinado em disco
modelo_treinado.to_disk("modelo_reserva")

# Testar com uma nova mensagem após o treinamento
modelo_reserva = spacy.load("modelo_reserva")

# Exemplo de teste
mensagem_teste = "Gostaria de reservar uma mesa dia 10 às 19:30."
doc = modelo_reserva(mensagem_teste)

# Exibir as entidades reconhecidas
for ent in doc.ents:
    print(f"Entidade: {ent.text}, Tipo: {ent.label_}")
