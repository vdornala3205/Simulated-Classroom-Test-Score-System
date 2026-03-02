#Simulated Classroom Test Score System

#TASK 1 - Generate and Inspet the Data

import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate scores for 5 students (rows) and 4 subjects (columns)
scores = np.random.randint(50, 101, size=(5, 4))

# Extract score of the 3rd student (index 2) in the 2nd subject (index 1)
score_3_2 = scores[2, 1]

# All scores of the last 2 students
last_two_students = scores[-2:]

# First 3 students, subjects 2 and 3 only (indices 1 and 2)
subset_scores = scores[:3, 1:3]

print("*********** Output - Task 1 : Generate and Inspet the Data ********\n")
print("Full Scores Array:\n", scores)
print(f"\nScore of the 3rd student in 2nd subject: {score_3_2}")
print(f"\nScores of the last 2 students (all subjects):\n{last_two_students}")
print(f"\nScores for the first 3 students in subjects 2 and 3 only:\n{subset_scores}")
print("\n")

# TASK 2 - Analyze with Boradcasting

# Column-wise mean ((average score per subject), rounded to 2 decimal places
subject_means = np.round(scores.mean(axis=0),2)

# Add curve [5, 3, 7, 2] and clip at 100
curve = np.array([5, 3, 7, 2])
curved_scores = np.clip(scores + curve, a_min=None, a_max=100)

# Row-wise max of curved scores (axis 1). Best subject score per student
student_maxes = curved_scores.max(axis=1)

print("*********** Output - Task 2 : Analyze with Broadcasting ********\n")
print(f"Subject Means: {subject_means}")
print(f"\nCurved scores (Clipped at 100):\n{curved_scores}")
print(f"\nBest subject score per student: {student_maxes}")
print("\n")

#TASK 3 - Normalize and Identify 

# Min-Max Normalization per row
row_min = curved_scores.min(axis=1, keepdims=True)
row_max = curved_scores.max(axis=1, keepdims=True)

normalized_scores = (curved_scores - row_min) / (row_max - row_min)

# Identify index of the single highest value
flat_index = np.argmax(normalized_scores)
student_idx, subject_idx = np.unravel_index(flat_index, normalized_scores.shape)

# Boolean masking for scores > 90
high_performers = curved_scores[curved_scores > 90]

print("*********** Output - Task 3 : Normalize & Identify ********\n")
print(f"Normalized Scores:\n{normalized_scores.round(2)}")
print(f"\nHighest Value found at: Student {student_idx}, Subject {subject_idx}")
print(f"\nScores > 90: {high_performers}")
print("\n")
