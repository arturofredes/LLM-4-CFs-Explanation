import pandas as pd

# Original negative outcome
original = pd.DataFrame({
    'Status of existing checking account': ['< 0 DM'],
    'Duration': [48],
    'Credit history': ['critical account/other credits existing (not at this bank)'],
    'Purpose': ['car (used)'],
    'Credit amount': [6143],
    'Savings account/bonds': ['< 100 DM'],
    'Present employment since': ['>= 7 years'],
    'Installment rate in percentage of disposable income': [4],
    'Personal status and sex': ['female: divorced/separated/married'],
    'Other debtors / guarantors': ['none'],
    'Property': ['unknown/no property'],
    'Age': [58],
    'Other installment plans': ['stores'],
    'Housing': ['for free'],
    'Number of existing credits at this bank': [2],
    'Job': ['unskilled - resident'],
    'Number of people being liable to provide maintenance for': [1],
    'Telephone': ['none'],
    'Foreign worker': ['yes'],
    'class': [0]
})

# Counterfactuals
counterfactuals = pd.DataFrame({
    'Status of existing checking account': ['< 0 DM', 'no checking account', '< 0 DM', '0 <= ... < 200 DM', '< 0 DM'],
    'Duration': [24, 48, 19, 48, 29],
    'Credit history': ['critical account/other credits existing (not at this bank)', 
                       'critical account/other credits existing (not at this bank)', 
                       'critical account/other credits existing (not at this bank)', 
                       'critical account/other credits existing (not at this bank)', 
                       'critical account/other credits existing (not at this bank)'],
    'Purpose': ['car (used)', 'car (used)', 'car (used)', 'car (used)', 'car (used)'],
    'Credit amount': [6143, 6143, 1101, 6143, 6143],
    'Savings account/bonds': ['< 100 DM', '< 100 DM', '< 100 DM', '< 100 DM', '< 100 DM'],
    'Present employment since': ['>= 7 years', '>= 7 years', '>= 7 years', '4 <= ... < 7 years', '>= 7 years'],
    'Installment rate in percentage of disposable income': [4, 4, 4, 4, 4],
    'Personal status and sex': ['female: divorced/separated/married', 
                                'female: divorced/separated/married', 
                                'female: divorced/separated/married', 
                                'female: divorced/separated/married', 
                                'female: divorced/separated/married'],
    'Other debtors / guarantors': ['none', 'none', 'none', 'none', 'none'],
    'Property': ['unknown/no property', 'unknown/no property', 'unknown/no property', 'unknown/no property', 'unknown/no property'],
    'Age': [58, 58, 58, 58, 58],
    'Other installment plans': ['stores', 'stores', 'stores', 'stores', 'stores'],
    'Housing': ['for free', 'for free', 'for free', 'for free', 'for free'],
    'Number of existing credits at this bank': [2, 2, 2, 2, 2],
    'Job': ['unskilled - resident', 'unskilled - resident', 'unskilled - resident', 'unskilled - resident', 'unskilled - resident'],
    'Number of people being liable to provide maintenance for': [1, 1, 1, 1, 1],
    'Telephone': ['none', 'yes, registered under the customer\'s name', 'none', 'none', 'none'],
    'Foreign worker': ['yes', 'yes', 'yes', 'yes', 'yes'],
    'class': [1, 1, 1, 0, 1]
})

# Rule 1: Checking Account Status
rule1 = counterfactuals['Status of existing checking account'].isin(['no checking account', '0 <= ... < 200 DM']).sum()

# Rule 2: Duration of Credit
rule2 = counterfactuals['Duration'] < 48

# Rule 3: Credit Amount
rule3 = counterfactuals['Credit amount'] < 6143

# Rule 4: Telephone
rule4 = (counterfactuals['Telephone'] == 'yes, registered under the customer\'s name').sum()

# Rule 5: Class
rule5 = (counterfactuals['class'] == 1).sum()

# Print results
print("Number of counterfactuals following Rule 1 (Checking Account Status):", rule1)
print("Number of counterfactuals following Rule 2 (Duration of Credit):", rule2.sum())
print("Number of counterfactuals following Rule 3 (Credit Amount):", rule3.sum())
print("Number of counterfactuals following Rule 4 (Telephone):", rule4)
print("Number of counterfactuals following Rule 5 (Class):", rule5)