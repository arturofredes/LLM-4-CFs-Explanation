import re
import subprocess
import sys
import numpy as np
import pandas as pd

###################################################################################
#PROMPT PROCESSING
###################################################################################

def extract_code(text):
    start_code = text.find("```python") + len("```python\n")
    end_code = text.find("```", start_code)
    return text[start_code:end_code].strip()

def extract_text(text):
    start_code = text.find("python") 
    end_code = text.find("```", start_code)
    return text[:start_code].strip() 

def extract_rules(text):
    # Regular expression pattern to find the last occurrence of "1."
    pattern = r"1\.\s"
    
    # Find all matches in the text
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return ""
    
    # Get the position of the last "1." in the text
    last_match = matches[-1]
    start_position = last_match.start()
    
    # Extract everything from the last "1." onward
    last_list = text[start_position:]
    
    return last_list.strip()

def execute_code(code, timeout=10):
    with open("temp_code.py", "w") as f:
        f.write(code)
    try:
        result = subprocess.run([sys.executable, "temp_code.py"], capture_output=True, text=True, check=True, timeout=timeout)
        return result.stdout, False
    except subprocess.CalledProcessError as e:
        return e.stdout + e.stderr, True
    except subprocess.TimeoutExpired:
        return "Execution timed out.", True
    
def save_code(code, timeout=10):
    with open("temp_code.py", "w") as f:
        f.write(code)

def join_outputs_explanation(rules, results,explanations):
    final_string = ''
    for i in range(len(rules)):
        final_string = final_string + f'\n\nSystem{i+1}:' + '\nRules:\n' +rules[i]+ '\nResults:\n' +results[i]+ '\nExplanation:\n'+ explanations[i]
    return final_string

def join_outputs_example(rules, results,examples, classifications):
    final_string = ''
    for i in range(len(rules)):
        final_string = final_string + f'\n\nSystem{i+1}:' + '\nRules:\n' +rules[i]+ '\nResults:\n' +results[i]+ '\nExample:\n'+ examples[i]+ '\nClassification:'+ classifications[i]
    return final_string

def join_outputs_eval(rules, results):
    final_string = ''
    for i in range(len(rules)):
        final_string = final_string + f'\n\nSystem{i+1}:' + '\nRules:\n' +rules[i]+ '\nResults:\n' +results[i]
    return final_string

def extract_number(s):
    try:
        # Attempt to convert the string to a float after stripping whitespace and newlines
        number = float(s.strip())
        return number
    except ValueError:
        # If conversion fails, return nan
        return np.nan 
    
def count_lines(input_string):
    # Split the string into lines
    lines = input_string.splitlines()

    # Count the number of lines
    return len(lines)

###################################################################################
#PROCESSING RESULTS
###################################################################################

def is_in_dataset(original_df, row_df):
    """
    Check if a given single-row DataFrame exists in another DataFrame.
    
    Parameters:
    - original_df (pd.DataFrame): The original dataset.
    - row_df (pd.DataFrame): A single-row DataFrame to check.
    
    Returns:
    - bool: True if the row exists in the original dataset, False otherwise.
    """
    # Merge the original dataset with the row dataframe based on all columns
    merged_df = pd.merge(original_df, row_df, how='inner', on=row_df.columns.tolist())
    
    # If the merged dataframe is not empty, the row exists in the original dataset
    return not merged_df.empty

def get_metrics(df,prompt,cfs):
    validity = df[f'{cfs}_cfs_{prompt}_label'].mean()
    rules = df[f'{cfs}_cfs_{prompt}_rules'].mean()
    rules_ratio = np.mean(df[f'{cfs}_cfs_{prompt}_rules_followed']/df[f'{cfs}_cfs_{prompt}_rules'])
    in_data = df[f'{cfs}_cfs_{prompt}_in_data'].mean()
    fail = df[f'{cfs}_cfs_{prompt}_status'].mean()
    first = df[f'{cfs}_cfs_{prompt}_rule_1'].mean()
    second = df[f'{cfs}_cfs_{prompt}_rule_2'].mean()
    third = df[f'{cfs}_cfs_{prompt}_rule_3'].mean()
    return [validity, rules, rules_ratio, in_data, first, second, third, fail]

def process_eval_csv(df):
    df = df.sort_values(['Importance'],ascending=False)
    n_rules = df.shape[0]
    rules_followed = df['In explanation'].sum()

    first_rule = 0
    second_rule = 0
    third_rule = 0
    #Check if df contains at least three rules
    if df.shape[0] >2:
        if df['In explanation'].iloc[0] == 1:
            first_rule = 1
        if df['In explanation'].iloc[1] == 1:
            second_rule = 1
        if df['In explanation'].iloc[2] == 1:
            third_rule = 1
    if df.shape[0]==2:
        if df['In explanation'].iloc[0] == 1:
            first_rule = 1
        if df['In explanation'].iloc[1] == 1:
            second_rule = 1
        third_rule = np.nan
    if df.shape[0]==1:
        if df['In explanation'].iloc[0] == 1:
            first_rule = 1
        second_rule = np.nan
        third_rule = np.nan
    return n_rules, rules_followed, first_rule, second_rule,third_rule


###################################################################################
#OTHERS
###################################################################################
def construct_variable(variable_list, info, i):
    return variable_list[i] + ": " + info[variable_list[i]]

def string_info(variable_list, info):
    string = ''
    for i in range(len(variable_list)):
        string += construct_variable(variable_list,info,i) + '\n'
    return string


def get_german_credit_data_info():
    """Feature descriptions for the Statlog German Credit (translated labels) dataset."""
    return {
        "Status of existing checking account": "Balance of the applicant's checking account at the bank.",
        "Duration": "Loan duration in months.",
        "Credit history": "Past credit behavior (paid duly, delays, critical account, etc.).",
        "Purpose": "Purpose of the credit (car, furniture, business, education, etc.).",
        "Credit amount": "Amount of the credit in DM.",
        "Savings account/bonds": "Savings account balance or bonds.",
        "Present employment since": "Time in current employment.",
        "Installment rate in percentage of disposable income": "Installment payments as a percentage of disposable income.",
        "Personal status and sex": "Personal status and sex category.",
        "Other debtors / guarantors": "Whether there are co-applicants or guarantors.",
        "Present residence since": "Years at current address.",
        "Property": "Most valuable property owned (real estate, car, unknown, etc.).",
        "Age": "Age in years.",
        "Other installment plans": "Other installment plans from banks or stores.",
        "Housing": "Housing situation (rent, own, for free).",
        "Number of existing credits at this bank": "Number of existing credits at this bank.",
        "Job": "Employment skill level / job type.",
        "Number of people being liable to provide maintenance for": "Number of dependents the applicant must support.",
        "Telephone": "Whether the customer is listed under their name in the telephone directory.",
        "Foreign worker": "Whether the applicant is a foreign worker.",
        "class": "Credit risk label: 1 = good credit, 0 = bad credit.",
    }


def get_loan_data_info():
    """Feature descriptions for the loan approval dataset (Home Credit / Loan prediction style)."""
    return {
        "Gender": "Applicant gender.",
        "Married": "Whether the applicant is married.",
        "Dependents": "Number of dependents (0, 1, 2, 3+ encoded as in the training data).",
        "Education": "Education level (Graduate / Not Graduate).",
        "Self_Employed": "Whether the applicant is self-employed.",
        "ApplicantIncome": "Applicant's monthly income.",
        "CoapplicantIncome": "Co-applicant's monthly income (0 if none).",
        "LoanAmount": "Requested loan amount (in thousands, as in the source data).",
        "Loan_Amount_Term": "Loan repayment term in months.",
        "Credit_History": "Credit history meets guidelines: 1 = yes, 0 = no / unknown.",
        "Property_Area": "Property location type (Urban, Semiurban, Rural).",
        "class": "Loan decision label: 1 = approved, 0 = denied.",
    }
