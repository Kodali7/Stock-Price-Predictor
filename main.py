from static.helper import *


def train_test_model(STOCK_NAME, day):
  ticker = yf.Ticker(STOCK_NAME)

  # Week_Start = date.today() - timedelta(days=7)
  # Week_Start.strftime('%Y-%m-%d')
  # Week_End = date.today() + timedelta(days=2)
  # Week_End.strftime('%Y-%m-%d')
  # week_dataframe = pd.DataFrame(yf.download(STOCK_NAME,start=Week_Start,end=Week_End))

  midday_data = ticker.history(period='1d',interval='5m')
  week_data = ticker.history(period='3mo', interval='5d')

  tz = midday_data.index.tzinfo
  currenttime = datetime.now(tz)

  day = input("Open or Close? ")
  # Formatting data
  scaler = MinMaxScaler(feature_range=(-1,1))
  scaled_data = scaler.fit_transform(week_data[day].values.reshape(-1,1))
  scaled_data_day = scaler.fit_transform(midday_data[day].values.reshape(-1,1))

  # Preparing data for model
  def prepare_data(data, length):
      sequence = []
      target = []
      for i in range(len(data)-length):
          sequence.append(data[i:i+length])
          target.append(data[i+length])
      return sequence, target
  if day=='Day':
      sequence,target = prepare_data(scaled_data_day,3)
  else:
    sequence, target = prepare_data(scaled_data,3)
  # Splitting data
  X_train, X_test, y_train, y_test = train_test_split(sequence, target, test_size=0.2,shuffle=False)

  # Training time
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  X_train, y_train = torch.tensor(X_train,dtype=torch.float32).to(device), torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).expand(-1,3,1).to(device)
  X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).expand(-1,3,1).to(device)

  model = LSTM_Model(input_shape=1,
                    hidden_layers=50,
                    layers=1,
                    output_shape=1).to(device)

  epochs=1000

  # Loss function
  loss_fn = nn.MSELoss()
  # Optimizer function
  optimizer = torch.optim.Adam(params=model.parameters(),
                              lr=0.001)

  for epoch in tqdm(range(epochs)):
      model.train()
      pred = model(X_train)
      loss = loss_fn(pred, y_train)
      acc = accuracy_fn(y_true=y_train,
                            y_pred=pred)

      optimizer.zero_grad()

      loss.backward()

      optimizer.step()

      model.eval()
      with torch.inference_mode():
        test_pred = model(X_test)
        test_loss = loss_fn(test_pred,y_test)
        test_acc = accuracy_fn(y_true=y_test,
                              y_pred=test_pred)
      if epoch % 100 ==0:
        print(f"Loss: {loss:.3f} | Test Loss: {test_loss:.3f}")

  # Finalizing for visualizing
  torch.save(model.state_dict(), 'model.pth')

  model.eval()
  with torch.inference_mode():
    final_sample = torch.tensor(scaled_data_day, dtype=torch.float32).unsqueeze(1).expand(-1,3,1).to(device)
    preds = model(final_sample)
  preds=scaler.inverse_transform(preds.squeeze())
  return preds
