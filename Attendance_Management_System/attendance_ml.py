import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load CSV file
data = pd.read_csv("attendance.csv")

# Calculate Attendance Percentage
data["Attendance_Percentage"] = (
    data["Classes_Attended"] /
    data["Total_Classes"]
) * 100

# Create Status Column
data["Status"] = data["Attendance_Percentage"].apply(
    lambda x: 1 if x >= 75 else 0
)

# Features and Target
X = data[["Attendance_Percentage"]]
y = data["Status"]

# Train Model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict Student Status
new_attendance = [[70]]

prediction = model.predict(new_attendance)

print("Prediction Result:")

if prediction[0] == 1:
    print("Good Attendance")
else:
    print("Low Attendance")

# Display Attendance Table
print("\nStudent Attendance Report")
print(data)

# Create Graph
plt.bar(data["Student"], data["Attendance_Percentage"])
plt.title("Student Attendance Percentage")
plt.xlabel("Student Name")
plt.ylabel("Attendance Percentage")
plt.show()
