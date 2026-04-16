from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os
import google.generativeai as genai

app = Flask(__name__)

# Try to configure Gemini
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    chat_model = genai.GenerativeModel('gemini-1.5-flash')
else:
    chat_model = None
# Load dataset
raw_df = pd.read_csv("gym_members_exercise_tracking.csv")
raw_df["BMI"] = raw_df["Weight (kg)"] / (raw_df["Height (m)"] ** 2)

# Define simple recommendation plan logic
def assign_plan(row):
    bmi = row["BMI"]
    level = row["Experience_Level"]

    if bmi > 25:
        return "Fat Loss Plan"
    if bmi < 18.5:
        return "Muscle Gain Plan"
    if level == "Beginner":
        return "Full Body"
    if level == "Intermediate":
        return "Upper Lower Split"
    return "Push Pull Legs"

raw_df["Plan"] = raw_df.apply(assign_plan, axis=1)

# Encode categorical values
raw_df["Gender"] = raw_df["Gender"].map({"Male": 1, "Female": 0})
raw_df["Experience_Level"] = raw_df["Experience_Level"].map({
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2,
})

raw_df["Workout_Type"] = raw_df["Workout_Type"].astype("category")
workout_type_map = {category: code for code, category in enumerate(raw_df["Workout_Type"].cat.categories)}
raw_df["Workout_Type"] = raw_df["Workout_Type"].cat.codes

X = raw_df.drop("Plan", axis=1)
y = raw_df["Plan"]

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

plan_info = {
    "Fat Loss Plan": [
        "4-5 weekly sessions with interval cardio and full-body lifts.",
        "Maintain a small calorie deficit and prioritize sleep.",
        "Track protein intake and stay hydrated throughout the day.",
    ],
    "Muscle Gain Plan": [
        "Use compound lifts and progressive overload 3-4 times per week.",
        "Eat a protein-rich diet and allow proper recovery.",
        "Focus on perfect form before increasing weight.",
    ],
    "Full Body": [
        "Beginner-friendly full-body sessions three times a week.",
        "Mix strength, mobility, and light cardio in each workout.",
        "Keep the volume manageable and build consistency first.",
    ],
    "Upper Lower Split": [
        "Alternate upper-body and lower-body training across four days.",
        "Give each muscle group enough recovery time.",
        "Use moderate volume to build balanced strength.",
    ],
    "Push Pull Legs": [
        "Train push, pull, and leg movements on separate days.",
        "Best for intermediate and advanced users with steady recovery.",
        "Balance volume and intensity for muscle growth.",
    ],
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    required_fields = ["age", "gender", "weight", "height", "workout_type", "level", "days", "goal"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing input fields."}), 400

    age = int(data["age"])
    gender = int(data["gender"])
    weight = float(data["weight"])
    height = float(data["height"])
    workout_type = data["workout_type"]
    level = int(data["level"])
    days = int(data["days"])
    goal = data["goal"]

    bmi = weight / (height ** 2)
    fat_percentage = 22 if bmi < 25 else 28 if bmi < 30 else 34
    max_bpm = max(140, 220 - age)
    avg_bpm = int(max_bpm * 0.72)
    resting_bpm = 68 if age < 40 else 72
    calories_burned = int(220 + weight * 4)
    water_intake = round(1.8 + (days - 3) * 0.3, 1)

    encoded_workout = workout_type_map.get(workout_type, 0)

    user = pd.DataFrame([
        {
            "Age": age,
            "Gender": gender,
            "Weight (kg)": weight,
            "Height (m)": height,
            "BMI": bmi,
            "Workout_Type": encoded_workout,
            "Experience_Level": level,
            "Workout_Frequency (days/week)": days,
            "Fat_Percentage": fat_percentage,
            "Max_BPM": max_bpm,
            "Avg_BPM": avg_bpm,
            "Resting_BPM": resting_bpm,
            "Session_Duration (hours)": 1.0,
            "Calories_Burned": calories_burned,
            "Water_Intake (liters)": water_intake,
        }
    ])

    user = user[X.columns]
    plan = model.predict(user)[0]

    if goal == "weight_loss" and plan != "Fat Loss Plan":
        plan = "Fat Loss Plan"
    elif goal == "muscle_gain" and plan == "Fat Loss Plan":
        plan = "Muscle Gain Plan"

    return jsonify({
        "plan": plan,
        "details": plan_info.get(plan, []),
        "metrics": {
            "bmi": round(bmi, 1),
            "fat_percentage": fat_percentage,
            "max_bpm": max_bpm,
            "avg_bpm": avg_bpm,
            "resting_bpm": resting_bpm,
            "calories_burned": calories_burned,
            "water_intake": water_intake
        }
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    message = data.get("message", "")
    
    if not message:
         return jsonify({"reply": "No message provided."}), 400

    if not chat_model:
         return jsonify({"reply": "We are currently operating offline. To enable the AI coach, add GEMINI_API_KEY environment variable."})

    try:
         prompt = f"You are a professional fitness coach acting as an AI assistant in 'FitForge'. Answer the user briefly, concisely, and politely. User: {message}"
         response = chat_model.generate_content(prompt)
         return jsonify({"reply": response.text})
    except Exception as e:
         return jsonify({"reply": f"Error contacting AI: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
