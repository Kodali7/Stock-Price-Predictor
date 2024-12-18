function submitForm() {
  let ticker = document.getElementById("ticker").value;
  let time = document.getElementById("time").value;
  let price = document.getElementById("price").value;

  fetch("https://localhost:5000/predictions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ticker: ticker, time: time, price: price }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Prediction:", data.prediction);
    })
    .catch((error) => {
      console.log("There was an error with the data processing");
      console.log("Error:", error);
    });
}
