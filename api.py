from flask import Flask, request, jsonify
import pickle
import pandas as pd
from word2number import w2n

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (the model should be in the 'model' folder)
model = pickle.load(open('model/salary_predictor.pkl', 'rb'))

# Create a prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_salary():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features from the request and make sure they match the DataFrame column names
        experience = data['experience']
        test_score = data['test_score']  # Match with the exact column name
        interview_score = data['interview_score']  # Match with the exact column name

        # Convert experience to number if it's in word format
        if isinstance(experience, str):
            experience = w2n.word_to_num(experience)

        # Create a DataFrame to pass to the model (with the exact column names)
        input_data = pd.DataFrame([[experience, test_score, interview_score]],
                                  columns=['experience', 'test_score(out of 10)', 'interview_score(out of 10)'])

        # Predict the salary
        salary = model.predict(input_data)[0]
        
        # Return the predicted salary in JSON format
        return jsonify({'predicted_salary': salary})

    except Exception as e:
        # Return an error response if something goes wrong
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
