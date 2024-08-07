from libraries import *

# Visualizing
def week_price_plotter(name,
                       price_type,
                       img,
                       dataframe,
                       predictions=None):
    plt.figure(figsize=(16,8))
    ticker_plot = dataframe
    plt.plot(ticker_plot.index, dataframe[price_type], label=f"{price_type.capitalize()} Price")
    plt.title(name)
    plt.ylabel('Close Price')
    plt.xlabel('Date')
    if predictions is not None:
      plt.plot(dataframe.index[:len(predictions)], predictions[:,0], color='red')
      plt.legend(['Actual Price', 'Predicted Price'])
    plt.savefig(img, format='png')
    plt.close()
def day_price_plotter(name,
                      price_type,
                      img,
                      dataframe,
                      predictions=None):
    plt.figure(figsize=(16,8))
    plt.plot(dataframe.index, dataframe[price_type], label=f"{price_type.capitalize()} Price")
    plt.title(name)
    plt.ylabel('Price')
    plt.grid(True)
    plt.xlabel('Time')
    if predictions is not None:
      plt.plot(dataframe.index[:len(predictions)], predictions[:,0], color='red')
      plt.legend(['Actual Price', 'Predicted Price'])
    plt.savefig(img, format='png')
    plt.close()

# Model-related
class LSTM_Model(nn.Module):
    def __init__(self,
                 input_shape:int,
                 hidden_layers:int,
                 layers:int,
                 output_shape:int):
        super(LSTM_Model, self).__init__()
        self.main = nn.LSTM(input_shape,
                    hidden_layers,
                    layers,
                    dropout=0.5)
        
        self.end = nn.Linear(in_features=hidden_layers,out_features=output_shape)

    def forward(self, x):
        self.out, self.extra = self.main(x)
        return self.end(self.out)

# Accuracy function
def accuracy_fn(y_true, y_pred):
    """
    Calculates accuracy between truth labels and predictions using Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (torch.Tensor): True labels for predictions.
        y_pred (torch.Tensor): Predicted values.

    Returns:
        float: Overall accuracy score as a percentage.
    """
    # Ensure tensors are on CPU and convert to numpy arrays
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    # Calculate MAPE for each stock prediction
    mape_scores = []
    for k in range(y_true.shape[1]):  # Assuming y_true and y_pred have the same shape
        actual = y_true[:, k]
        predicted = y_pred[:, k]
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        mape_scores.append(mape)

    # Calculate overall accuracy as the average of MAPE scores
    overall_accuracy = np.mean(mape_scores)

    return overall_accuracy

