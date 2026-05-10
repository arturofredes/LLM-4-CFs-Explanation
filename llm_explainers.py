from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import pandas as pd
import dice_ml
from pathlib import Path
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load variables from a .env file in the project root (this file's directory).
load_dotenv(Path(__file__).resolve().parent / ".env")

from prompts import *
from utils import *


def _resolve_google_api_key(explicit: Optional[str] = None) -> str:
    """API key for Gemini Developer API (AI Studio), from argument or env."""
    for candidate in (
        (explicit or "").strip(),
        (os.environ.get("GOOGLE_API_KEY") or "").strip(),
        (os.environ.get("GEMINI_API_KEY") or "").strip(),
    ):
        if candidate:
            return candidate
    return ""


def _build_chat_llm(
    llm_provider: str,
    model_name: str,
    *,
    google_api_key: Optional[str] = None,
):
    """Return a LangChain chat model for OpenAI or Google Gemini."""
    provider = (llm_provider or "openai").lower().strip()
    if provider == "openai":
        return ChatOpenAI(model=model_name)
    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError(
                "Gemini support requires: pip install langchain-google-genai google-generativeai"
            ) from e
        key = _resolve_google_api_key(google_api_key)
        if not key:
            raise ValueError(
                "Gemini requires an API key: set GOOGLE_API_KEY or GEMINI_API_KEY in the "
                "environment (e.g. in `.env` for docker compose `env_file`), or pass "
                "google_api_key when constructing the explainer."
            )
        # Ensure the Google stack sees the key (avoids gRPC falling through to ADC when
        # only the constructor argument was set). See langchain-google-genai #1271.
        os.environ["GOOGLE_API_KEY"] = key

        kwargs = {"model": model_name, "google_api_key": key}
        model_fields = getattr(ChatGoogleGenerativeAI, "model_fields", None)
        if model_fields and "transport" in model_fields:
            kwargs["transport"] = "rest"
        return ChatGoogleGenerativeAI(**kwargs)
    raise ValueError(
        f"Unsupported llm_provider={llm_provider!r}; use 'openai' or 'gemini'."
    )


class LLMExplanation4CFs:
    def __init__(
        self,
        model,
        backend: str,
        model_description: str,
        dataset_info: str,
        continuous_features,
        outcome_name: str,
        training_set: pd.DataFrame,
        test_set: pd.DataFrame,
        llm="gemini-3.1-flash-lite",
        llm_provider: str = "gemini",
        google_api_key: Optional[str] = None,
        prompt_type="zero",
        n_counterfactuals=5,
        user_input=False,
        positive_prediction_message: Optional[str] = None,
        # DiCE generate_counterfactuals options (method "random" ignores genetic-only weights)
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, Any]] = None,
        stopping_threshold: float = 0.5,
        posthoc_sparsity_param: float = 0.1,
        posthoc_sparsity_algorithm: str = "linear",
        dice_random_seed: Optional[int] = None,
        dice_sample_size: Optional[int] = None,
    ) -> None:
        self.model = model                                #Load the model we want to explain
        self.backend = backend                            # brief explanation of the ML model
        self.model_description = model_description        # Framework used to build the model (used to generate counterfactuals)
        self.dataset_info = dataset_info                  # string information about the dataset
        self.continuous_features = continuous_features    # List of continuous features (Necessary for the counterfactual generation)
        self.outcome_name = outcome_name                  # Label (Necessary for counterfactual generation)
        self.positive_prediction_message = (
            positive_prediction_message
            or "This person is predicted to earn more than 50k$"
        )
        self.training = training_set                      # Necessary for counterfactual generation
        self.test = test_set                              # Necessary to  check novelty of the evaluation example
        self.llm = llm                                    # LLM used, works with Langchain
        self.llm_provider = llm_provider                  # 'openai' or 'gemini'
        self.google_api_key = google_api_key              # optional; else GOOGLE_API_KEY env
        self.prompt_type = prompt_type                    # zero or one
        self.n_counterfactuals = n_counterfactuals        # Number of counterfactuals used in the explanation 
        self.user_input = user_input                      # Human in the loop helping select the causes
        self.features_to_vary = features_to_vary
        self.permitted_range = permitted_range
        self.stopping_threshold = stopping_threshold
        self.posthoc_sparsity_param = posthoc_sparsity_param
        if posthoc_sparsity_algorithm not in ("linear", "binary"):
            raise ValueError(
                "posthoc_sparsity_algorithm must be 'linear' or 'binary' "
                f"(got {posthoc_sparsity_algorithm!r})"
            )
        self.posthoc_sparsity_algorithm = posthoc_sparsity_algorithm
        self.dice_random_seed = dice_random_seed
        self.dice_sample_size = dice_sample_size

    def _dice_cf_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for DiCE ``generate_counterfactuals`` (except query and total_CFs)."""
        out: Dict[str, Any] = {
            "stopping_threshold": self.stopping_threshold,
            "posthoc_sparsity_param": self.posthoc_sparsity_param,
            "posthoc_sparsity_algorithm": self.posthoc_sparsity_algorithm,
        }
        fv = self.features_to_vary
        if fv is None or len(fv) == 0:
            out["features_to_vary"] = "all"
        else:
            out["features_to_vary"] = list(fv)
        if self.permitted_range:
            out["permitted_range"] = dict(self.permitted_range)
        if self.dice_random_seed is not None:
            out["random_seed"] = int(self.dice_random_seed)
        if self.dice_sample_size is not None:
            out["sample_size"] = int(self.dice_sample_size)
        return out

    def fit(self):
        # Fit the counterfactual generation class
        # Step 1: dice_ml.Data
        d = dice_ml.Data(dataframe=self.training, continuous_features=self.continuous_features, outcome_name=self.outcome_name)
        # Using sklearn backend
        m = dice_ml.Model(model=self.model, backend=self.backend)
        # Using method=random for generating CFs
        self.exp = dice_ml.Dice(d, m, method="random", )


        # Create the different chains that will be used
        llm = _build_chat_llm(
            self.llm_provider,
            self.llm,
            google_api_key=self.google_api_key,
        )

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
        pred = int(pd.Series(self.model.predict(user_data)).iloc[0])
        if pred == 1:
            return self.positive_prediction_message
        
        #generate counterfactuals
        counterfactuals = self.exp.generate_counterfactuals(
            user_data[0:1],
            total_CFs=self.n_counterfactuals,
            desired_class="opposite",
            **self._dice_cf_kwargs(),
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
        pred = int(pd.Series(self.model.predict(user_data)).iloc[0])
        if pred == 1:
            return self.positive_prediction_message
        
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
        counterfactuals = self.exp.generate_counterfactuals(
            user_data[0:1],
            total_CFs=self.n_counterfactuals,
            desired_class="opposite",
            **self._dice_cf_kwargs(),
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
        if self.outcome_name in final_cf.columns:
            final_cf.drop(columns=[self.outcome_name], inplace=True)

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
    def __init__(
        self,
        model,
        backend: str,
        model_description: str,
        dataset_info: str,
        continuous_features,
        outcome_name: str,
        training_set: pd.DataFrame,
        test_set,
        llm="gemini-3.1-flash-lite",
        llm_provider: str = "gemini",
        google_api_key: Optional[str] = None,
        prompt_type="zero",
        n_counterfactuals=5,
        branches=3,
        user_input=False,
        positive_prediction_message: Optional[str] = None,
        features_to_vary: Optional[List[str]] = None,
        permitted_range: Optional[Dict[str, Any]] = None,
        stopping_threshold: float = 0.5,
        posthoc_sparsity_param: float = 0.1,
        posthoc_sparsity_algorithm: str = "linear",
        dice_random_seed: Optional[int] = None,
        dice_sample_size: Optional[int] = None,
    ) -> None:
        self.model = model                                #Load the model we want to explain
        self.backend = backend                            # brief explanation of the ML model
        self.model_description = model_description        # Framework used to build the model (used to generate counterfactuals)
        self.dataset_info = dataset_info                  # string information about the dataset
        self.continuous_features = continuous_features    # List of continuous features (Necessary for the counterfactual generation)
        self.outcome_name = outcome_name                  # Label (Necessary for counterfactual generation)
        self.positive_prediction_message = (
            positive_prediction_message
            or "This person is predicted to earn more than 50k$"
        )
        self.training = training_set                      # Necessary for counterfactual generation
        self.test = test_set                              # Necessary to  check novelty of the evaluation example
        self.llm = llm                                    # LLM used, works with Langchain
        self.llm_provider = llm_provider
        self.google_api_key = google_api_key
        self.prompt_type = prompt_type                    # zero or one
        self.n_counterfactuals = n_counterfactuals        # Number of counterfactuals used in the explanation 
        self.user_input = user_input                      # Human in the loop helping select the causes
        self.branches = branches       
        self.features_to_vary = features_to_vary
        self.permitted_range = permitted_range
        self.stopping_threshold = stopping_threshold
        self.posthoc_sparsity_param = posthoc_sparsity_param
        if posthoc_sparsity_algorithm not in ("linear", "binary"):
            raise ValueError(
                "posthoc_sparsity_algorithm must be 'linear' or 'binary' "
                f"(got {posthoc_sparsity_algorithm!r})"
            )
        self.posthoc_sparsity_algorithm = posthoc_sparsity_algorithm
        self.dice_random_seed = dice_random_seed
        self.dice_sample_size = dice_sample_size

    def _nested_cf_kwargs(self) -> Dict[str, Any]:
        return dict(
            features_to_vary=self.features_to_vary,
            permitted_range=self.permitted_range,
            stopping_threshold=self.stopping_threshold,
            posthoc_sparsity_param=self.posthoc_sparsity_param,
            posthoc_sparsity_algorithm=self.posthoc_sparsity_algorithm,
            dice_random_seed=self.dice_random_seed,
            dice_sample_size=self.dice_sample_size,
        )

    def fit(self):
        nk = self._nested_cf_kwargs()
        self.exp_m1 = LLMExplanation4CFs(model = self.model, #Load the model we want to explain
                            model_description = self.model_description, # brief explanation of the ML model
                            backend = self.backend, # Framework used to build the model (used to generate counterfactuals)
                            dataset_info=self.dataset_info , # string information about the dataset
                            continuous_features= self.continuous_features, # Necessary for the counterfactual generation
                            outcome_name = self.outcome_name, #Necessary for counterfactual generation
                            training_set = self.training, #Necessary for counterfactual generation
                            test_set = self.test, #Necessary to  check novelty of the evaluation example
                            llm = self.llm, #LLM used, works with Langchain 
                            llm_provider=self.llm_provider,
                            google_api_key=self.google_api_key,
                            prompt_type = 'zero', # zero or one
                            n_counterfactuals = self.n_counterfactuals, #Number of counterfactuals used in the explanation 
                            user_input = False, #Human in the loop helping select the causes
                            positive_prediction_message=self.positive_prediction_message,
                            **nk,
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
                            llm_provider=self.llm_provider,
                            google_api_key=self.google_api_key,
                            prompt_type = 'one', # zero or one
                            n_counterfactuals = self.n_counterfactuals, #Number of counterfactuals used in the explanation 
                            user_input = False, #Human in the loop helping select the causes
                            positive_prediction_message=self.positive_prediction_message,
                            **nk,
                           )

        self.exp_m1.fit()
        self.exp_m2.fit()

        llm = _build_chat_llm(
            self.llm_provider,
            self.llm,
            google_api_key=self.google_api_key,
        )

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
        if self.outcome_name in final_cf.columns:
            final_cf.drop(columns=[self.outcome_name], inplace=True)

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

