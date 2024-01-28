# libraries
import pandas as pd
import joblib
from flask import Flask, render_template_string, request

app = Flask(__name__)

# HTML şablonu
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Music Submission Form</title>
<style>
  body {
    font-family: Arial, sans-serif;
    background-color: {{ background_color }};
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;
    margin: 0;
  }

  .container {
    background-color: white;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    text-align: center;
  }

  .input-group {
    margin: 10px 0;
  }

  input[type="text"] {
    border: 2px solid #ccc;
    border-radius: 5px;
    padding: 10px;
    width: 80%;
    margin-bottom: 20px;
  }

  input[type="submit"] {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    transition-duration: 0.4s;
    cursor: pointer;
    border-radius: 5px;
  }

  input[type="submit"]:hover {
    background-color: #45a049;
  }
</style>
</head>
<body>

<div class="container">
  <h2>Write a song!</h2>
  <form action="/" method="post">
    <div class="input-group">
      <input type="text" id="color" name="color" placeholder="Write a song!">
    </div>
    <div class="input-group">
      <input type="submit" value="Submit">
    </div>
  </form>
</div>

</body>
</html>
"""

#our model and data table
path = 'data/spotify_data.csv'
df = pd.read_csv(path)

df.dropna(inplace=True)

# train and test
# Create DataFrame X by dropping specified columns
X_withTracknames = df.drop(columns=["artist_name", "track_id", "genre"])
X = df.drop(columns=["artist_name", "track_name", "track_id", "genre"])

# Load the saved model from the file
loaded_model1 = joblib.load('trained_model_energy.pkl')
loaded_model2 = joblib.load('trained_model_liveness.pkl')
loaded_model3 = joblib.load('trained_model_tempo.pkl')


def rgb_to_hex(r, g, b):
    # Ensure the RGB values are within the valid range (0-255)
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))

    # Convert RGB to hexadecimal
    hex_code = "#{:02x}{:02x}{:02x}".format(r, g, b)

    return hex_code


@app.route('/', methods=['GET', 'POST'])
def change_background_color():
    background_color = '#f0f0f0'  # Default background color
    if request.method == 'POST':
        songName = request.form['color']
        #remove spaces
        songName = songName.strip()

        indices = X_withTracknames.index[df.isin([songName]).any(axis=1)].tolist()

        new_data = X.iloc[indices]

        prediction1 = loaded_model1.predict(new_data)
        prediction2 = loaded_model2.predict(new_data)
        prediction3 = loaded_model3.predict(new_data)

        print("Prediction1:", prediction1)
        print("Prediction2:", prediction2)
        print("Prediction3:", prediction3)

        # Get the color code from the form
        background_color = rgb_to_hex(prediction2[0], prediction1[0], prediction3[0])
        #background_color = request.form['color']

    return render_template_string(HTML_TEMPLATE, background_color=background_color)



if __name__ == '__main__':
    app.run(debug=True)
