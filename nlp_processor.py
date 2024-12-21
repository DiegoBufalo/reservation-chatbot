import spacy

# Carregar o modelo de NLP para português
nlp = spacy.load("modelo_reserva")

def processar_mensagem(mensagem):
    """
    Função para processar a mensagem do usuário e extrair informações de reserva.
    """
    doc = nlp(mensagem)
    
    # Imprimir todas as entidades reconhecidas pelo spaCy para depuração
    print("Entidades encontradas:", [(ent.text, ent.label_) for ent in doc.ents])
    
    # Exemplo: Vamos tentar identificar "data", "horário" e "quantidade de pessoas"
    data = None
    horario = None
    num_pessoas = None
    
    for ent in doc.ents:
        if ent.label_ == "DATA":
            data = ent.text
        elif ent.label_ == "HORARIO":
            horario = ent.text
        elif ent.label_ == "NUM_PESSOAS":
            num_pessoas = int(ent.text)  # Convertendo para inteiro
    
    return {"data": data, "horario": horario, "num_pessoas": num_pessoas}
