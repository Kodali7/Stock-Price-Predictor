from flask import Flask, request, Response, jsonify, send_file
from flask_cors import CORS, cross_origin
from main import *

app = Flask(__name__)
app.debug = True
# CORS(app, supports_credentials=True, resources={r"/predictions": {"origins": "http://127.0.0.1:5500"}})
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model = LSTM_Model(input_shape=1,
                   hidden_layers=50,
                   layers=2,
                   output_shape=1).to(device)
dir_path = '/Users/saikodali/Documents/GitHub/Stock-Price-Predictor'
file_path = os.path.join(dir_path, 'model.pth')

@app.route('/predictions', methods=['POST','OPTIONS'])
@cross_origin(origins='*', headers=['Content-Type'])
def stockPredict():
    if request.method == "OPTIONS":
        return build_cors_preflight_response()
    if request.method == "POST":
        try:
            print(request.json, "SOMETHING NEW")
            request_data = request.json
            ticker_symbol = request_data.get('ticker')  # Get ticker symbol as string
            time = request_data.get('time')
            price_type = request_data.get('price')

            ticker = yf.Ticker(ticker_symbol)  # Create ticker object
            midday_data = ticker.history(period='1d',interval='5m')
            week_data = ticker.history(period='3mo', interval='5d')
            print("HERE BEFORE TRAINING")
            if (time == 'Day'):
                print('STRING')
            preds = train_test_model(ticker_symbol, time)  # something wrong with time
            print("HERE AFTER TRAINING")
            model.load_state_dict(torch.load(file_path,map_location=device,weights_only=True))
            model.eval()

            img = BytesIO()
            if time == 'Day':
                # print("WE ARE IN THE THING") #debugging
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
            response = send_file(img,mimetype='image/png', as_attachment=False)
            response.headers.add("Access-Control-Allow-Origin", "*")
            return response

        except Exception as e:
            app.logger.error(f"An error occurred: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return corsify_actual_response(jsonify({'Error': 'Method not allowed'})), 405
        
def build_cors_preflight_response():
    response = Response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == "__main__":
    app.run(debug=True)