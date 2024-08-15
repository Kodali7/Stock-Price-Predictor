from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS, cross_origin
from main import *

app = Flask(__name__)
# CORS(app, supports_credentials=True, resources={r"/predictions": {"origins": "http://127.0.0.1:5500"}})
# app.config['CORS_ALLOW_HEADERS'] = 'Content-Type'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTM_Model(input_shape=1,
                   hidden_layers=50,
                   layers=1,
                   output_shape=1).to(device)
# FIX THIS LINE
dir_path = '/Users/saikodali/Documents/GitHub/Stock-Price-Predictor'
file_path = os.path.join(dir_path, 'model.pth')
with open(file_path, 'w') as file:
    ...
model.load_state_dict(torch.load(file_path, map_location=device, weights_only=True))
# checkpoint = torch.load('model.pt', map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

@app.route('/predictions', methods=['OPTIONS'])
@cross_origin()
def intercept():
    response = make_response()
    open('worked', 'w')
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

@app.route('/predictions', methods=['POST'])
# @cross_origin()
def stockPredict():
    try:
        request_data = request.json
        ticker = request_data.get('ticker','')
        time = request_data.get('time', '')
        price_type = request_data.get('price','')

        midday_data = ticker.history(period='1d',interval='5m')
        week_data = ticker.history(period='3mo', interval='5d')

        preds = train_test_model(ticker, time)

        img = BytesIO()
        if time == 'Day':
            day_price_plotter(ticker,
                              price_type,
                              img,
                              midday_data,
                              preds)
        else:
            week_price_plotter(ticker,
                               price_type,
                               img,
                               week_data,
                               preds)
        img.seek(0)
        return send_file(img,mimetype='image/png',as_attachment=False, attachment_filename='plot.png')

    except Exception as e:
        return jsonify({'Error:', str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)