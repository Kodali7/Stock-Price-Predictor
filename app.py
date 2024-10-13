from flask import Flask, request, Response, jsonify, send_file
from flask_cors import CORS, cross_origin
from main import *

app = Flask(__name__)
app.debug = True
# CORS(app, supports_credentials=True, resources={r"/predictions": {"origins": "http://127.0.0.1:5500"}})
CORS(app, resources={r"/predictions": {"origins": "http://127.0.0.1:5500"}})
app.config['CORS_ALLOW_HEADERS'] = 'Content-Type'


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model = LSTM_Model(input_shape=1,
                   hidden_layers=50,
                   layers=2,
                   output_shape=1).to(device)
# FIX THIS LINE
dir_path = '/Users/saikodali/Documents/GitHub/Stock-Price-Predictor'
file_path = os.path.join(dir_path, 'model.pth')

# checkpoint = torch.load('model.pt', map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])


@app.route('/predictions', methods=['POST','OPTIONS'])
@cross_origin(origin='*', allow_headers=['Content-Type'])
def stockPredict():
    # if request.method == "OPTIONS":
    #     return _build_cors_preflight_response()
    if request.method == "POST":
        try:
            print(request.json, "SOMETHING NEW")
            request_data = request.json
            ticker = yf.Ticker(request_data.get('ticker'))
            time = request_data.get('time')
            price_type = request_data.get('price')

            midday_data = ticker.history(period='1d',interval='5m')
            week_data = ticker.history(period='3mo', interval='5d')

            preds = train_test_model(ticker, time)
            # Loading the data from model.pth
            model.load_state_dict(torch.load(file_path,map_location=device,weights_only=True))
            model.eval()

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
            response =  send_file(img,mimetype='image/png', as_attachment=False)
            response.headers["Access-Control-Allow-Origin"] = "*"
            _corsify_actual_response(response)
            print(response)
            return response

        except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return _corsify_actual_response(jsonify({'Error': 'Method not allowed'})), 405
        
def _build_cors_preflight_response():
    response = Response()
    response.headers.add("Access-Control-Allow-Origin", "http://127.0.0.1:5500")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "http://127.0.0.1:5500")
    return response

if __name__ == "__main__":
    app.run(debug=True)
    stockPredict()