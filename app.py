from flask import Flask, request, jsonify
from nlp_processor import processar_mensagem
import requests

app = Flask(__name__)

# Exemplo de endpoint para enviar a reserva para a sua API SaaS
def realizar_reserva(data, horario, num_pessoas):
    url_api = "https://sua-api-saas.com/reservas"
    
    payload = {
        "data": data,
        "horario": horario,
        "num_pessoas": num_pessoas
    }
    
    response = requests.post(url_api, json=payload)
    return response.json()

@app.route("/chatbot", methods=["POST"])
def chatbot(): 
    dados_usuario = request.json
    mensagem = dados_usuario.get("mensagem", "")
    
    # Processar a mensagem para extrair informações de reserva
    dados_reserva = processar_mensagem(mensagem)
    
    # Verificar se as informações essenciais estão presentes
    # if not dados_reserva["data"] or not dados_reserva["horario"] or not dados_reserva["num_pessoas"]:
    #     return jsonify({"erro": "Informações insuficientes para realizar a reserva"}), 400
    
    # Realizar a reserva
    # resultado_reserva = realizar_reserva(
    #     dados_reserva["data"],
    #     dados_reserva["horario"],
    #     dados_reserva["num_pessoas"]
    # )

    payload = {
        "data": dados_reserva["data"],
        "horario": dados_reserva["horario"],
        "num_pessoas": dados_reserva["num_pessoas"]
    }
    
    return jsonify(payload)

if __name__ == "__main__":
    app.run(debug=True)
