from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and encoder
model = pickle.load(open("fare_model.pkl", "rb"))  # Load trained model
encoder = pickle.load(open("encoder.pkl", "rb"))  # Load encoder

@app.route("/", methods=["GET", "POST"])  # ✅ Fix: Allow both GET & POST
def home():
    if request.method == "POST":
        try:
            # Extract form data
            distance = float(request.form["distance"])
            traffic = request.form["traffic"]
            time_of_day = request.form["time_of_day"]

            # Prepare input DataFrame
            input_data = pd.DataFrame([[distance, traffic, time_of_day]], columns=["distance", "Traffic", "Time_of_Day"])

            # One-hot encode categorical features
            input_encoded = encoder.transform(input_data[["Traffic", "Time_of_Day"]])
            input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(["Traffic", "Time_of_Day"]))

            # Merge numerical & encoded categorical features
            final_input = pd.concat([input_data[["distance"]], input_encoded_df], axis=1)

            # Make prediction
            prediction = model.predict(final_input)

            # ✅ Fix: Format fare to 2 decimal places
            formatted_fare = f"{prediction[0]:.2f}"

            return render_template("index.html", prediction_text=f"Estimated Fare: $ {formatted_fare}")

        except Exception as e:
            return render_template("index.html", prediction_text="Error in prediction. Check inputs.")

    return render_template("index.html")  # Handle GET request

if __name__ == "__main__":
    app.run(debug=True)
