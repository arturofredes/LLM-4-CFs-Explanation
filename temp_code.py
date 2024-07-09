import pandas as pd

# Load the positive example from temp_csv.csv
df = pd.read_csv('temp_csv.csv')

# Define the rules and the number of counterfactuals following each rule from the systems
rules = [
    {"Rule": "Education Upgrade", "Importance": 2 + 2 + 1, "Condition": lambda row: row['education'] in ['Masters', 'Doctorate', 'Bachelors']},
    {"Rule": "Consideration of Marital Status", "Importance": 3 + 4, "Condition": lambda row: row['marital_status'] in ['Divorced', 'Married']},
    {"Rule": "Flexibility in Working Hours", "Importance": 5 + 1, "Condition": lambda row: row['hours_per_week'] != 40},
    {"Rule": "Professional and White-Collar Occupations", "Importance": 3, "Condition": lambda row: row['occupation'] in ['Professional', 'White-Collar']},
    {"Rule": "Increase Working Hours to at Least 74", "Importance": 1, "Condition": lambda row: row['hours_per_week'] >= 74},
    {"Rule": "Gender Change to Male", "Importance": 1, "Condition": lambda row: row['gender'] == 'Male'},
    {"Rule": "Change Occupation to Service", "Importance": 1, "Condition": lambda row: row['occupation'] == 'Service'}
]

# Check which rules the example follows
results = []
for rule in rules:
    follows_rule = int(rule["Condition"](df.iloc[0]))
    results.append({
        "Rule": rule["Rule"],
        "Importance": rule["Importance"],
        "In explanation": follows_rule
    })

# Create the final dataframe
df_final = pd.DataFrame(results)

# Save to csv
df_final.to_csv('evaluation.csv', index=False)