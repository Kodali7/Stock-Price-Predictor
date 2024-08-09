from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from main import *

app = Flask(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTM_Model(input_shape=1,
                   hidden_layers=50,
                   layers=1,
                   output_shape=1).to(device)
# FIX THIS LINE
# model.load_state_dict(torch.load('model.pth', map_location=device, weight_only=True))
# checkpoint = torch.load('model.pt', map_location='cpu')
# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


@app.route('/prediction', methods=['POST'])
@cross_origin()
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