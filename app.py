from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    if request.method == "POST":
        
        # Recebendo dados do formulário
        idade = int(request.form.get("idade"))
        sexo = bool(int(request.form.get("sexo")))
        filhos = int(request.form.get("filhos"))
        altura = int(request.form.get("altura")) / 100
        peso = float(request.form.get("peso"))
        fumante = float(request.form.get("fumante"))
        regiao = float(request.form.get("regiao"))

        # Calculando o IMC
        imc = peso / (altura ** 2)

        # Preparando os dados para o modelo
        pessoa = pd.DataFrame([{
            "age": idade,
            "sex": sexo,
            "bmi": imc,
            "children": filhos,
            "smoker": fumante,
            "region": regiao
        }])

        # Carregando o modelo e fazendo a previsão
        model = joblib.load("python-model/04_Insurance_LinReg_ECMD.pkl")
        predicted = model.predict(pessoa)
        value = predicted[0][0]

        # Formatando o resultado para enviar para o template
        resultado = {
            'idade': idade,
            'sexo': 'Masculino' if sexo else 'Feminino',
            'altura': f'{altura * 100:.1f} cm',
            'peso': f'{peso:.1f} kg',
            'imc': f'{imc:.1f}',
            'classificacao': f'{value:.2f}'
        }

    # Enviando para o template
    return render_template("index.html", resultado=resultado)


if __name__ == "__main__":
    app.run(debug=True)