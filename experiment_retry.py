import pickle 
import pandas as pd
from dice_ml.utils import helpers
from prompts import *
from utils import *
from llm_explainers import *
from pathlib import Path
import os

directory = Path("./experiments")
# Create the directory if it does not exist
directory.mkdir(parents=True, exist_ok=True)
prompt = input('prompt strategy (zero, one, ToT): ')
cf = int(input('n counterfactuals: '))
prev_exp = pd.read_csv(f'./experiments/{prompt}/{cf}_cfs/experiment_{prompt}_{cf}_cfs.csv')
test100 = prev_exp[prev_exp[f'{cf}_cfs_{prompt}_status'] == 1]
print(test100.shape[0],'experiments failed and will be retried')

test100.reset_index(inplace=True)
test100.rename(columns={'index': 'previous_index'}, inplace=True)


# Load the model we want to explain
with open("""./models/loan_model.pkl""", 'rb') as file:
    model = pickle.load(file)

#Load the train and test data set
train_dataset = pd.read_csv('./data/adult_train_dataset.csv')
test_dataset = pd.read_csv('./data/adult_test_dataset.csv')


columns = ['age', 'workclass', 'education', 'marital_status', 'occupation', 'race',
       'gender', 'hours_per_week']
model_description = """ML-system that predicts wether a person will earn more than 50k $ a year"""

directory = Path(f"./experiments/{prompt}")
# Create the directory if it does not exist
directory.mkdir(parents=True, exist_ok=True)

directory = Path(f"./experiments/{prompt}/{cf}_cfs/evals/")
# Create the directory if it does not exist
directory.mkdir(parents=True, exist_ok=True)
directory = Path(f"./experiments/{prompt}/{cf}_cfs/examples/")
# Create the directory if it does not exist
directory.mkdir(parents=True, exist_ok=True)
if prompt == 'ToT':
    n_branches = int(input('Number of branches: '))
    exp_m = ToTLLMExplanation4CFs(model = model, #Load the model we want to explain
                            model_description = """ML-system that predicts wether a person will earn more than 50k $ a year""", # brief explanation of the ML model
                            backend='sklearn', # Framework used to build the model (used to generate counterfactuals)
                            dataset_info=string_info(train_dataset.columns, helpers.get_adult_data_info()) , # string information about the dataset
                            continuous_features=['age', 'hours_per_week'], # Necessary for the counterfactual generation
                            outcome_name= 'income', #Necessary for counterfactual generation
                            training_set=train_dataset, #Necessary for counterfactual generation
                            test_set= test_dataset, #Necessary to  check novelty of the evaluation example
                            llm='gpt-4o', #LLM used, works with Langchain
                            prompt_type='one', # zero or one
                            n_counterfactuals=cf, #Number of counterfactuals used in the explanation 
                            user_input=False, #Human in the loop helping select the causes
                            branches = 3
                        )
else:
    exp_m = LLMExplanation4CFs(model = model, #Load the model we want to explain
                            model_description = """ML-system that predicts wether a person will earn more than 50k $ a year""", # brief explanation of the ML model
                            backend='sklearn', # Framework used to build the model (used to generate counterfactuals)
                            dataset_info=string_info(train_dataset.columns, helpers.get_adult_data_info()) , # string information about the dataset
                            continuous_features=['age', 'hours_per_week'], # Necessary for the counterfactual generation
                            outcome_name= 'income', #Necessary for counterfactual generation
                            training_set=train_dataset, #Necessary for counterfactual generation
                            test_set= test_dataset, #Necessary to  check novelty of the evaluation example
                            llm='gpt-4o', #LLM used, works with Langchain
                            prompt_type=prompt, # zero or one
                            n_counterfactuals=cf, #Number of counterfactuals used in the explanation 
                            user_input=False #Human in the loop helping select the causes
                        )



exp_m.fit()
for i in range(test100.shape[0]):
    ind = test100.loc[i,'previous_index']
    print(ind)
    file1 = f'./experiments/{prompt}/{cf}_cfs/examples/ex_{cf}cfs_{ind}.csv'
    file2 = f'./experiments/{prompt}/{cf}_cfs/evals/eval_{cf}cfs_{ind}.csv'
    # Check if file1 exists and remove it if it does
    if os.path.exists(file1):
        os.remove(file1)
    if os.path.exists(file2):
        os.remove(file2)
    try:              
        example_label, n_rules, rules_followed, first_rule, second_rule,third_rule, in_dataset = exp_m.explain_evaluate(user_data = pd.DataFrame(test100.loc[i, columns]).transpose(), verbose = False, return_all=False)
        os.rename('./temp_files/temp_csv.csv', f'./experiments/{prompt}/{cf}_cfs/examples/ex_{cf}cfs_{ind}.csv')
        os.rename('./temp_files/evaluation.csv', f'./experiments/{prompt}/{cf}_cfs/evals/eval_{cf}cfs_{ind}.csv')            
        prev_exp.loc[ind, str(cf) + '_cfs_' + prompt + '_label'] = example_label
        prev_exp.loc[ind, str(cf) + '_cfs_' +prompt + '_rules'] = n_rules
        prev_exp.loc[ind, str(cf) + '_cfs_' +prompt + '_rules_followed'] = rules_followed
        prev_exp.loc[ind, str(cf) + '_cfs_' + prompt + '_rule_1'] = first_rule
        prev_exp.loc[ind, str(cf) + '_cfs_' + prompt + '_rule_2'] = second_rule
        prev_exp.loc[ind, str(cf) + '_cfs_' + prompt + '_rule_3'] = third_rule
        prev_exp.loc[ind, str(cf) + '_cfs_' +prompt + '_in_data'] = in_dataset
        prev_exp.loc[ind, str(cf) + '_cfs_' +prompt + '_status'] = 0
        print('success')
    except Exception as e:
        print(e)
        prev_exp.loc[ind, str(cf) + '_cfs_' +prompt + '_status'] = 1
#merge
prev_exp.to_csv(f'./experiments/{prompt}/{cf}_cfs/experiment_{prompt}_{cf}_cfs.csv')
prediction, n_rules, rules_followed,in_data, first, second, third, fail = get_metrics(prev_exp,prompt,cf)
print('Prediction of the generated example: ', prediction)
print('Number of rules generated: ', n_rules)
print('Number of rules followed: ', rules_followed)
print('1st rule followed ', first)
print('2nd rule followed ', second)
print('3rd rule followed ', third)
print('Example exists in data: ', in_data) 
      