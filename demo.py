import streamlit as st
import pickle
import pandas as pd
import dice_ml
from dice_ml.utils import helpers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import subprocess
import sys
import re 

from llm_explainers import *

st.set_page_config(layout="wide")
with open("""./models/loan_model.pkl""", 'rb') as file:
    model = pickle.load(file)
train_dataset = pd.read_csv('./data/adult_train_dataset.csv')
test_dataset = pd.read_csv('./data/adult_test_dataset.csv')
model_description = """ML-system that predicts wether a person will earn more than 50k $ a year"""

# HTML and CSS to create a green box for the markdown
st.markdown("""
<style>
.green-box {
    background-color: #90ee90;  /* Light green background */
    border: 1px solid green;
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
.yellow-box {
    background-color: #ffff99;  /* Light yellow background */
    border: 1px solid #ffcc00;  /* Darker yellow border */
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.red-box {
    background-color: #ffcccc;  /* Light red background */
    border: 1px solid #ff3333;  /* Darker red border */
    padding: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)
################################################################################################33
def main(): 
    if 'data' not in st.session_state:
        data = {
                    'age': [28],
                    'workclass': ['Other/Unknown'],
                    'education': ['Assoc'],
                    'marital_status': ['Separated'],
                    'occupation': ['Other/Unknown'],
                    'race': ['White'],
                    'gender': ['Female'],
                    'hours_per_week': [40],
                }

        st.session_state['data'] = pd.DataFrame(data)
    if 'parameters' not in st.session_state:
        st.session_state['parameters'] = 'ok'
        st.session_state['prompt_type'] = 'zero'
        st.session_state['n_cfs'] = 5
        

    st.title('Income Predictor 💵')
    with st.expander('User Data'):
        with st.form("data_form"):
            age = st.text_input('Age', value = 28)
            workclass = st.selectbox("Work class", ('Private', 'Self-Employed', 'Other/Unknown', 'Government'), index = 2)
            education = st.selectbox("Education", ('Bachelors', 'Assoc', 'Some-college', 'School', 'HS-grad',
            'Masters', 'Prof-school', 'Doctorate'), index = 1)
            marital_status = st.selectbox("Marital status", ('Single', 'Married', 'Divorced', 'Widowed', 'Separated'), index = 4)
            occupation = st.selectbox('Occupation', ('White-Collar', 'Professional', 'Service', 'Blue-Collar',
            'Other/Unknown', 'Sales'), index = 4)
            race = st.selectbox("Race", ('White', 'Other'), index = 0)
            gender = st.selectbox("Gender", ('Female', 'Male'))
            hours_per_week = st.text_input('Hours per week', value = 40)


            submitted_data = st.form_submit_button("Save Data", use_container_width = True)
            if submitted_data:
                data = {
                    'age': [int(age)],
                    'workclass': [workclass],
                    'education': [education],
                    'marital_status': [marital_status],
                    'occupation': [occupation],
                    'race': [race],
                    'gender': [gender],
                    'hours_per_week': [int(hours_per_week)],
                }

                st.session_state['data'] = pd.DataFrame(data)

    
    with st.expander('Parameters'):
        with st.form("param_form"):
            llm = st.selectbox("Select an LLM", ('gpt-3.5', 'gpt-4','gpt-4o'), index = 2)
            prompt_type = st.selectbox("Prompt", ('Zero Shot', 'One Shot', 'Tree of Thought (ToT)'), index = 0)
            n_cfs = st.slider('# of counterfactuals generated', min_value=1, max_value=5, value=5, step=2)
            
            submitted_parameters = st.form_submit_button("Save Parameters", use_container_width = True)
            if submitted_parameters:
                st.session_state['parameters'] = 'ok'
                if prompt_type == 'Zero Shot':
                    st.session_state['prompt_type'] = 'zero'
                if prompt_type == 'One Shot':
                    st.session_state['prompt_type'] = 'one'
                else :
                    st.session_state['prompt_type'] = 'tot'
                st.session_state['n_cfs'] = n_cfs
                st.session_state['llm'] = llm

    with st.form("explain_form"):
        submitted_explain = st.form_submit_button("Explain Case", use_container_width = True)
        if submitted_explain:
            st.subheader('Data')
            st.dataframe(st.session_state['data'])
            classif = model.predict(st.session_state['data'])
            if classif == 1:
                st.markdown('<div class="green-box">'+'This user is predicted to earn over 50k$ a year'+'</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="red-box">'+'This user is predicted to earn less than 50k$ a year'+'</div>', unsafe_allow_html=True)
                print(st.session_state['prompt_type'])



            if st.session_state['prompt_type'] == 'tot':
                exp_m = ToTLLMExplanation4CFs(model = model, #Load the model we want to explain
                        model_description = """ML-system that predicts wether a person will earn more than 50k $ a year""", # brief explanation of the ML model
                        backend='sklearn', # Framework used to build the model (used to generate counterfactuals)
                        dataset_info=string_info(train_dataset.columns, helpers.get_adult_data_info()) , # string information about the dataset
                        continuous_features=['age', 'hours_per_week'], # Necessary for the counterfactual generation
                        outcome_name= 'income', #Necessary for counterfactual generation
                        training_set=train_dataset, #Necessary for counterfactual generation
                        test_set= test_dataset, #Necessary to  check novelty of the evaluation example
                        llm='gpt-4o', #LLM used, works with Langchain
                        prompt_type='zero', # zero or one
                        n_counterfactuals=st.session_state['n_cfs'], #Number of counterfactuals used in the explanation 
                        user_input=False, #Human in the loop helping select the causes
                        branches = 3
                    )
                exp_m.fit()

                out, explanation, code2, final_cf, code3, final_df, prediction, n_rules,rules_followed, first_rule, second_rule,third_rule, in_data = exp_m.explain_evaluate(user_data = st.session_state['data'], verbose = False,return_all=True) 
            
                with st.expander('Combined output'):
                    st.markdown(out)
                with st.expander('Explanation'):
                    st.markdown(explanation)
                with st.expander('Evaluation'):
                    st.subheader('Generating Example')
                    st.code(code2,'python')
                    st.dataframe(final_df)
                    st.subheader('Evaluating example')
                    st.code(code3,'python')
                    st.text('Predicted class: '+str(prediction))
                    st.text('Rules followed by the example: '+ str(rules_followed))
                    st.text('The example is in the original data: ' + str(in_data))
                    st.text('The example follows 1st ranked rule: ' + str(first_rule))
                    st.text('The example follows 2nd ranked rule: ' + str(second_rule))
                    st.text('The example follows 3rd ranked rule: ' + str(third_rule))
            
            
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
                                prompt_type=st.session_state['prompt_type'], # zero or one
                                n_counterfactuals=st.session_state['n_cfs'], #Number of counterfactuals used in the explanation 
                                user_input=False #Human in the loop helping select the causes
                            )
                
                exp_m.fit()
                counterfactuals, rules, code1, result1, explanation, code2, final_cf, code3, prediction, n_rules,rules_followed, first_rule, second_rule,third_rule,  is_in_data = exp_m.explain_evaluate(user_data = st.session_state['data'], verbose = False,return_all=True)
                with st.expander('Counterfactuals'):
                    st.dataframe(counterfactuals)
                with st.expander('Identified Causes'):
                    st.markdown(rules)
                with st.expander('Checking and Ranking Causes'):
                    st.subheader('Code')
                    st.code(code1,'python')
                    st.subheader('Execution results')
                    st.text(result1)
                with st.expander('Explanation'):
                    st.markdown(explanation)
                with st.expander('Evaluation'):
                    st.subheader('Generating Example')
                    st.code(code2,'python')
                    st.dataframe(final_cf)
                    st.subheader('Evaluating example')
                    st.code(code3,'python')
                    st.text('Predicted class: '+str(prediction))
                    st.text('Rules followed by the example: '+ str(rules_followed))
                    st.text('The example is in the original data: ' + str(is_in_data))
                    st.text('The example follows 1st ranked rule: ' + str(first_rule))
                    st.text('The example follows 2nd ranked rule: ' + str(second_rule))
                    st.text('The example follows 3rd ranked rule: ' + str(third_rule))

main()