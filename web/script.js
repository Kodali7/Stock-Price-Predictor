function submitForm() {
  let ticker = document.getElementById("ticker").value;
  let time = document.getElementById("time").value;
  let price = document.getElementById("price").value;

  fetch("http://127.0.0.1:5000/predictions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ticker: ticker, time: time, price: price }),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Prediction:", data);
    })
    .then((response) => {
      if (response.ok) {
        return response.blob(); 
      } else {
        throw new Error("Network response was not ok.");
      }
    })
    .then((imageBlob) => {
      console.log("Creating a image blob");
      
      const imageUrl = URL.createObjectURL(imageBlob);

      document.getElementById("predictedImage").src = imageUrl;
    })
    .catch((error) => {
      console.log("There was an error with the data processing");
      console.log("Error:", error);
    });
}
