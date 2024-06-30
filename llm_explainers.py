from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
import pandas as pd
import dice_ml
from pathlib import Path
from prompts import *
from utils import *
import os

class LLMExplanation4CFs:
    def __init__(self, model, backend: str,model_description: str, dataset_info: str, continuous_features, outcome_name: str, training_set: pd.DataFrame, test_set: pd.DataFrame, llm = 'gpt-4o', prompt_type = 'zero', n_counterfactuals=5, user_input=False) -> None:
        self.model = model                                #Load the model we want to explain
        self.backend = backend                            # brief explanation of the ML model
        self.model_description = model_description        # Framework used to build the model (used to generate counterfactuals)
        self.dataset_info = dataset_info                  # string information about the dataset
        self.continuous_features = continuous_features    # List of continuous features (Necessary for the counterfactual generation)
        self.outcome_name = outcome_name                  # Label (Necessary for counterfactual generation)
        self.training = training_set                      # Necessary for counterfactual generation
        self.test = test_set                              # Necessary to  check novelty of the evaluation example
        self.llm = llm                                    # LLM used, works with Langchain
        self.prompt_type = prompt_type                    # zero or one
        self.n_counterfactuals = n_counterfactuals        # Number of counterfactuals used in the explanation 
        self.user_input = user_input                      # Human in the loop helping select the causes
                           

    def fit(self):
        # Fit the counterfactual generation class
        # Step 1: dice_ml.Data
        d = dice_ml.Data(dataframe=self.training, continuous_features=self.continuous_features, outcome_name=self.outcome_name)
        # Using sklearn backend
        m = dice_ml.Model(model=self.model, backend=self.backend)
        # Using method=random for generating CFs
        self.exp = dice_ml.Dice(d, m, method="random", )


        # Create the different chains that will be used
        llm = ChatOpenAI(model_name=self.llm)

        if self.prompt_type == 'zero':
            template1 = ZeroShotRules()
            template2 = ZeroShotRulesCode()
            template3 = ZeroShotExplanation(self.user_input)
            template4 = ZeroShotExample()
            template5 = ZeroShotExampleCode()

        elif self.prompt_type == 'one':
            template1 = OneShotRules()
            template2 = OneShotRulesCode()
            template3 = OneShotExplanation(self.user_input)
            template4 = OneShotExample()
            template5 = OneShotExampleCode()

        else:
            raise ValueError('Not a prompting strategy considered')

        prompt = PromptTemplate(
            input_variables=["ML-system", "negative_outcome","positive_outcome"],
            template=template1,
        )
        self.rules_chain = LLMChain(llm=llm, prompt=prompt)


        prompt = PromptTemplate(
            input_variables=["ML-system", "negative_outcome","positive_outcome",'rules',"dataset_info"],
            template=template2,
        )
        self.rulescode_chain = LLMChain(llm=llm, prompt=prompt)

        if self.user_input:
                vars = ["ML-system", "negative_outcome","rules",'results','user_input',"dataset_info"]
        else:
                vars = ["ML-system", "negative_outcome","rules",'results',"dataset_info"]

        prompt = PromptTemplate(
            input_variables = vars,
            template=template3,
        )

        self.explanation_chain = LLMChain(llm=llm, prompt=prompt)


        vars = ["ML-system", "negative_outcome","explanation","dataset_info"]

        prompt = PromptTemplate(
            input_variables = vars,
            template=template4,
        )

        self.example_chain = LLMChain(llm=llm, prompt=prompt)

        prompt = PromptTemplate(
            input_variables=["ML-system", "negative_outcome","rules","results","dataset_info"],
            template=template5,
        )

        self.examplecode_chain = LLMChain(llm=llm, prompt=prompt)

    def explain(self, user_data: pd.DataFrame, verbose = False , return_all = False):
        if user_data.shape[0] != 1:
            return 'Error, you should provide a single example'
        pred = self.model.predict(user_data)
        if pred == 1:
            return 'This person is predicted to earn more than 50k$'
        
        #generate counterfactuals
        counterfactuals = self.exp.generate_counterfactuals(user_data[0:1], total_CFs=self.n_counterfactuals, desired_class="opposite",
                                                #features_to_vary=["workclass",'hours_per_week','occupation']
                                                )
        if verbose:
            print(counterfactuals._cf_examples_list[0].final_cfs_df)
        #generate rules and code
        response = self.rules_chain.run(({
        "ML-system": self.model_description,
        "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
        "positive_outcome": counterfactuals._cf_examples_list[0].final_cfs_df.to_string
        }))

        rules = response

        response = self.rulescode_chain.run(({
            "ML-system": self.model_description,
            "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
            "positive_outcome": counterfactuals._cf_examples_list[0].final_cfs_df.to_string,
            "rules" : rules,
            "dataset_info": self.dataset_info
            }))
        code = extract_code(response)
        if verbose == True:
            print(rules)
            print(code)
        result, _ = execute_code(code)
        if verbose == True:
            print(result)

        if self.user_input:
            selection = input('Select themost relevant rules'+'\nRules:\n'+ rules + '\nRelevance:\n' + result)
            explanation = self.explanation_chain.run(({
                "ML-system": self.model_description,
                "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
                "positive_outcome": counterfactuals._cf_examples_list[0].final_cfs_df.to_string,
                "rules": rules,
                "results": result,
                'user_input': selection,
                "dataset_info": self.dataset_info
                }))
        else: 
                explanation = self.explanation_chain.run(({
                "ML-system": self.model_description,
                "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
                "positive_outcome": counterfactuals._cf_examples_list[0].final_cfs_df.to_string,
                "rules": rules,
                "results": result,
                "dataset_info": self.dataset_info
                }))
        if return_all:
            return counterfactuals._cf_examples_list[0].final_cfs_df, rules, code, result, explanation
        else:
            return explanation
        
    def explain_evaluate(self, user_data: pd.DataFrame, verbose = False, return_all = False):
        if user_data.shape[0] != 1:
            return 'Error, you should provide a single example'
        pred = self.model.predict(user_data)
        if pred == 1:
            return 'This person is predicted to earn more than 50k$'
        
        directory = Path("./temp_files")
        # Create the directory if it does not exist
        directory.mkdir(parents=True, exist_ok=True)
        # Paths to the files
        file1 = './temp_csv.csv'
        file2 = './evaluation.csv'
        file3 = './temp_files/temp_csv.csv'
        file4 = './temp_files/evaluation.csv'
        # Check if file1 exists and remove it if it does
        if os.path.exists(file1):
            os.remove(file1)
        # Check if file2 exists and remove it if it does
        if os.path.exists(file2):
            os.remove(file2)
        # Check if file3 exists and remove it if it does
        if os.path.exists(file3):
            os.remove(file3)
        # Check if file4 exists and remove it if it does
        if os.path.exists(file4):
            os.remove(file4)
        #generate counterfactuals
        counterfactuals = self.exp.generate_counterfactuals(user_data[0:1], total_CFs=self.n_counterfactuals, desired_class="opposite",
                                                #features_to_vary=["workclass",'hours_per_week','occupation']
                                                )
        if verbose:
            print(counterfactuals._cf_examples_list[0].final_cfs_df)
        #generate rules and code
        response = self.rules_chain.run(({
        "ML-system": self.model_description,
        "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
        "positive_outcome": counterfactuals._cf_examples_list[0].final_cfs_df.to_string
        }))


        rules = response

        #Generate code to check the rules
        response = self.rulescode_chain.run(({
            "ML-system": self.model_description,
            "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
            "positive_outcome": counterfactuals._cf_examples_list[0].final_cfs_df.to_string,
            "rules" : rules,
            "dataset_info": self.dataset_info
            }))
        
        #Extract ad execute code
        code1 = extract_code(response)
        if verbose == True:
            print(rules)
            print(code1)
        result1, _ = execute_code(code1)
        if verbose == True:
            print(result1)

        #Generate the explanation
        if self.user_input:
            selection = input('Select themost relevant rules'+'\nRules:\n'+ rules + '\nRelevance:\n' + result1)
            explanation = self.explanation_chain.run(({
                "ML-system": self.model_description,
                "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
                "positive_outcome": counterfactuals._cf_examples_list[0].final_cfs_df.to_string,
                "rules": rules,
                "results": result1,
                'user_input': selection,
                "dataset_info": self.dataset_info
                }))
        else: 
                explanation = self.explanation_chain.run(({
                "ML-system": self.model_description,
                "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
                "positive_outcome": counterfactuals._cf_examples_list[0].final_cfs_df.to_string,
                "rules": rules,
                "results": result1,
                "dataset_info": self.dataset_info
                }))

        if verbose:
            print(explanation)

        #Generate example
        example = self.example_chain.run(({
            "ML-system": self.model_description,
            "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
            "explanation": explanation,
            "dataset_info": self.dataset_info
            }))
        code2 = extract_code(example)
        if verbose:
            print(code2)

        execute_code(code2)
        
        final_cf = pd.read_csv('./temp_csv.csv')
        if 'income' in final_cf.columns:
            final_cf.drop(columns=['income'], inplace = True)
        

        rule_check = self.examplecode_chain.run(({
            "ML-system": self.model_description,
            "negative_outcome": counterfactuals._cf_examples_list[0].test_instance_df.to_string,
            "rules": rules,
            "results" : result1,
            "dataset_info": self.dataset_info
            }))
        
        if verbose:
            print(rule_check)

            
        code3 = extract_code(rule_check)
        result3, _ = execute_code(code3)
        if verbose:
            print(code3)
        os.rename('temp_csv.csv', './temp_files/temp_csv.csv')
        os.rename('evaluation.csv', './temp_files/evaluation.csv')
        final_df = pd.read_csv('./temp_files/evaluation.csv')
        n_rules, rules_followed, first_rule, second_rule,third_rule = process_eval_csv(final_df)

        if return_all:
            return counterfactuals._cf_examples_list[0].final_cfs_df, rules, code1,  result1, explanation, code2, final_cf, code3, \
            self.model.predict(final_cf)[0], n_rules,rules_followed, first_rule, second_rule,third_rule, \
            is_in_dataset(pd.concat([self.training, self.test], axis=0).reset_index(drop=True),final_cf)
        else:
            return self.model.predict(final_cf)[0], n_rules,rules_followed, first_rule, second_rule,third_rule, \
                is_in_dataset(pd.concat([self.training, self.test], axis=0).reset_index(drop=True),final_cf)

class ToTLLMExplanation4CFs():
    def __init__(self, model, backend: str, model_description: str, dataset_info: str, continuous_features, outcome_name: str,training_set: pd.DataFrame, test_set, llm = 'gpt-4o', prompt_type = 'zero', n_counterfactuals=5, branches = 3, user_input = False) -> None:
        self.model = model                                #Load the model we want to explain
        self.backend = backend                            # brief explanation of the ML model
        self.model_description = model_description        # Framework used to build the model (used to generate counterfactuals)
        self.dataset_info = dataset_info                  # string information about the dataset
        self.continuous_features = continuous_features    # List of continuous features (Necessary for the counterfactual generation)
        self.outcome_name = outcome_name                  # Label (Necessary for counterfactual generation)
        self.training = training_set                      # Necessary for counterfactual generation
        self.test = test_set                              # Necessary to  check novelty of the evaluation example
        self.llm = llm                                    # LLM used, works with Langchain
        self.prompt_type = prompt_type                    # zero or one
        self.n_counterfactuals = n_counterfactuals        # Number of counterfactuals used in the explanation 
        self.user_input = user_input                      # Human in the loop helping select the causes
        self.branches = branches       


    def fit(self):
        self.exp_m1 = LLMExplanation4CFs(model = self.model, #Load the model we want to explain
                            model_description = self.model_description, # brief explanation of the ML model
                            backend = self.backend, # Framework used to build the model (used to generate counterfactuals)
                            dataset_info=self.dataset_info , # string information about the dataset
                            continuous_features= self.continuous_features, # Necessary for the counterfactual generation
                            outcome_name = self.outcome_name, #Necessary for counterfactual generation
                            training_set = self.training, #Necessary for counterfactual generation
                            test_set = self.test, #Necessary to  check novelty of the evaluation example
                            llm = self.llm, #LLM used, works with Langchain 
                            prompt_type = 'zero', # zero or one
                            n_counterfactuals = self.n_counterfactuals, #Number of counterfactuals used in the explanation 
                            user_input = False #Human in the loop helping select the causes
                           )

        self.exp_m2 = LLMExplanation4CFs(model = self.model, #Load the model we want to explain
                            model_description = self.model_description, # brief explanation of the ML model
                            backend = self.backend, # Framework used to build the model (used to generate counterfactuals)
                            dataset_info=self.dataset_info , # string information about the dataset
                            continuous_features= self.continuous_features, # Necessary for the counterfactual generation
                            outcome_name = self.outcome_name, #Necessary for counterfactual generation
                            training_set = self.training, #Necessary for counterfactual generation
                            test_set = self.test, #Necessary to  check novelty of the evaluation example
                            llm = self.llm, #LLM used, works with Langchain 
                            prompt_type = 'one', # zero or one
                            n_counterfactuals = self.n_counterfactuals, #Number of counterfactuals used in the explanation 
                            user_input = False #Human in the loop helping select the causes
                           )

        self.exp_m1.fit()
        self.exp_m2.fit()

        llm = ChatOpenAI(model_name="gpt-4o")

        template = ToT_explain()
        prompt = PromptTemplate(
            input_variables=["ML-system", 'negative_outcome',"output_list"],
            template=template,
        )
        self.explain_chain = LLMChain(llm=llm, prompt=prompt)

        if self.prompt_type == 'one':
            template = OneShotExample()
        else: 
            template = ZeroShotExample()
        prompt = PromptTemplate(
            input_variables=["ML-system", "negative_outcome","explanation","dataset_info"],
            template=template,
        )
        self.example_chain = LLMChain(llm=llm, prompt=prompt)


        template = ToT_ExampleCode()
        prompt = PromptTemplate(
            input_variables=["ML-system",'negative_outcome', "output_list",'dataset_info'],
            template=template,
        )
        self.example_eval_chain = LLMChain(llm=llm, prompt=prompt)

    def explain(self, user_data: pd.DataFrame, verbose = False, return_all = False ):
        explanations = []
        rules = []
        results = []
        for i in range(self.branches):
            if i%2 == 1:
                cfs, rule, code, result, explanation = self.exp_m2.explain(user_data, return_all=True)
            else:
                cfs, rule, code, result, explanation = self.exp_m1.explain(user_data, return_all=True)
            explanations.append(explanation)
            rules.append(rule)
            results.append(result)
        out = join_outputs_explanation(rules, results, explanations)


        response = self.explain_chain.run(({
        "ML-system": self.model_description,
        'negative_outcome': user_data.to_string,
        "output_list": out
        }))
        if return_all:
            return out, response
        else:
            return response

    def explain_evaluate(self, user_data: pd.DataFrame, verbose = False, return_all = False):

        directory = Path("./temp_files")
        # Create the directory if it does not exist
        directory.mkdir(parents=True, exist_ok=True)
        # Paths to the files
        file1 = './temp_csv.csv'
        file2 = './evaluation.csv'
        file3 = './temp_files/temp_csv.csv'
        file4 = './temp_files/evaluation.csv'
        # Check if file1 exists and remove it if it does
        if os.path.exists(file1):
            os.remove(file1)
        # Check if file2 exists and remove it if it does
        if os.path.exists(file2):
            os.remove(file2)
        # Check if file3 exists and remove it if it does
        if os.path.exists(file3):
            os.remove(file3)
        # Check if file4 exists and remove it if it does
        if os.path.exists(file4):
            os.remove(file4)
        explanations = []
        rules = []
        results = []


        #Generate explanation
        for i in range(self.branches):
            if i%2 == 1:
                cfs, rule, code, result, explanation = self.exp_m2.explain(user_data, return_all=True)
                
            else:
                cfs, rule, code, result, explanation = self.exp_m1.explain(user_data, return_all=True)         
            explanations.append(explanation)
            rules.append(rule)
            results.append(result)
        out = join_outputs_explanation(rules, results, explanations)
        out_eval = join_outputs_eval(rules, results)


        explanation = self.explain_chain.run(({
        "ML-system": self.model_description,
        'negative_outcome': user_data.to_string,
        "output_list": out
        }))


        # Counterfactual
        example = self.example_chain.run(({
            "ML-system": self.model_description,
            "negative_outcome": user_data.to_string,
            "explanation": explanation,
            "dataset_info": self.dataset_info
            }))
        code2 = extract_code(example)
        if verbose:
            print(code2)

        execute_code(code2)
        final_cf = pd.read_csv('temp_csv.csv')
        if 'income' in final_cf.columns:
            final_cf.drop(columns=['income'], inplace = True)

        #Evaluate the counterfactual
        rule_check = self.example_eval_chain.run(({
            "ML-system": self.model_description,
            "negative_outcome": user_data.to_string,
            "output_list" : out_eval,
            "dataset_info": self.dataset_info
            }))
        
        if verbose:
            print(rule_check)

            
        code3 = extract_code(rule_check)
        result3, _ = execute_code(code3)
        if verbose:
            print(code3)
        final_df = pd.read_csv('evaluation.csv')
        os.rename('temp_csv.csv', './temp_files/temp_csv.csv')
        os.rename('evaluation.csv', './temp_files/evaluation.csv')
        n_rules, rules_followed, first_rule, second_rule, third_rule = process_eval_csv(final_df)

        if return_all:
            return out, explanation,code2, final_cf, code3, final_df, self.model.predict(final_cf)[0], n_rules,rules_followed, first_rule, second_rule,third_rule, \
                is_in_dataset(pd.concat([self.training, self.test], axis=0).reset_index(drop=True),final_cf)            
        else:
            return self.model.predict(final_cf)[0], n_rules,rules_followed, first_rule, second_rule,third_rule, \
                is_in_dataset(pd.concat([self.training, self.test], axis=0).reset_index(drop=True),final_cf)

