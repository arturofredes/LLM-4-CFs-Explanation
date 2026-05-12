import json
import pickle

import pandas as pd
import streamlit as st
from dice_ml.utils import helpers

from llm_explainers import *

st.set_page_config(layout="wide")

DEFAULT_LLM_PROVIDER = "gemini"
DEFAULT_LLM_MODEL = "gemini-3.1-flash-lite"

GERMAN_CATEGORICAL_COLS = [
    "Status of existing checking account",
    "Credit history",
    "Purpose",
    "Savings account/bonds",
    "Present employment since",
    "Personal status and sex",
    "Other debtors / guarantors",
    "Property",
    "Other installment plans",
    "Housing",
    "Job",
    "Telephone",
    "Foreign worker",
]
GERMAN_CONTINUOUS_FEATURES = [
    "Duration",
    "Credit amount",
    "Installment rate in percentage of disposable income",
    "Age",
    "Present residence since",
    "Number of existing credits at this bank",
    "Number of people being liable to provide maintenance for",
]

LOAN_CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Education",
    "Self_Employed",
    "Credit_History",
    "Property_Area",
]
LOAN_CONTINUOUS_FEATURES = [
    "Dependents",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]


@st.cache_resource
def load_adult_bundle():
    with open("./models/adult_model.pkl", "rb") as file:
        model = pickle.load(file)
    train_dataset = pd.read_csv("./data/adult_train_dataset.csv")
    test_dataset = pd.read_csv("./data/adult_test_dataset.csv")
    dataset_info = string_info(
        train_dataset.columns, helpers.get_adult_data_info()
    )
    return {
        "key": "adult",
        "model": model,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "continuous_features": ["age", "hours_per_week"],
        "outcome_name": "income",
        "model_description": "ML-system that predicts whether a person will earn more than 50k $ a year",
        "dataset_info": dataset_info,
        "positive_prediction_message": "This person is predicted to earn more than 50k$ a year; counterfactual explanations are only generated for the negative class.",
        "page_title": "Income Predictor 💵",
    }


@st.cache_resource
def load_german_bundle():
    with open("./models/german_credit_model.pkl", "rb") as file:
        model = pickle.load(file)
    train_dataset = pd.read_csv("./data/germancredit_train_dataset.csv")
    test_dataset = pd.read_csv("./data/germancredit_test_dataset.csv")
    dataset_info = string_info(
        train_dataset.columns, get_german_credit_data_info()
    )
    cat_options = {
        c: sorted(train_dataset[c].astype(str).unique().tolist())
        for c in GERMAN_CATEGORICAL_COLS
    }
    return {
        "key": "german",
        "model": model,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "continuous_features": GERMAN_CONTINUOUS_FEATURES,
        "outcome_name": "class",
        "model_description": "ML-system that predicts whether a person has good or bad credit",
        "dataset_info": dataset_info,
        "positive_prediction_message": "This applicant is predicted to have good credit; counterfactual explanations are only generated for the negative class.",
        "page_title": "Credit risk predictor 🏦",
        "german_cat_options": cat_options,
    }


@st.cache_resource
def load_loan_bundle():
    with open("./models/loan_model.pkl", "rb") as file:
        model = pickle.load(file)
    train_dataset = pd.read_csv("./data/loan_train_dataset.csv")
    test_dataset = pd.read_csv("./data/loan_test_dataset.csv")
    dataset_info = string_info(train_dataset.columns, get_loan_data_info())
    cat_options = {
        c: sorted(train_dataset[c].astype(str).unique().tolist())
        for c in LOAN_CATEGORICAL_COLS
    }
    return {
        "key": "loan",
        "model": model,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "continuous_features": LOAN_CONTINUOUS_FEATURES,
        "outcome_name": "class",
        "model_description": "ML system that predicts whether a loan application is approved or denied",
        "dataset_info": dataset_info,
        "positive_prediction_message": "This applicant is predicted to have the loan approved; counterfactual explanations are only generated for the negative class.",
        "page_title": "Loan approval predictor 🏠",
        "loan_cat_options": cat_options,
    }


def default_adult_user_row():
    return pd.DataFrame(
        {
            "age": [28],
            "workclass": ["Other/Unknown"],
            "education": ["Assoc"],
            "marital_status": ["Separated"],
            "occupation": ["Other/Unknown"],
            "race": ["White"],
            "gender": ["Female"],
            "hours_per_week": [40],
        }
    )


def default_user_row(bundle):
    if bundle["key"] == "adult":
        return default_adult_user_row()
    train = bundle["train_dataset"]
    outcome = bundle["outcome_name"]
    return train.drop(columns=[outcome]).iloc[[0]].copy()


# HTML and CSS to create a green box for the markdown
st.markdown(
    """
<style>
.green-box {
    background-color: #90ee90;  /* Light green background */
    border: 1px solid green;
    padding: 10px;
    border-radius: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)


st.markdown(
    """
<style>
.yellow-box {
    background-color: #ffff99;  /* Light yellow background */
    border: 1px solid #ffcc00;  /* Darker yellow border */
    padding: 10px;
    border-radius: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<style>
.red-box {
    background-color: #ffcccc;  /* Light red background */
    border: 1px solid #ff3333;  /* Darker red border */
    padding: 10px;
    border-radius: 5px;
}
</style>
""",
    unsafe_allow_html=True,
)


def _render_explain_result(bundle, exp_m, prompt_type):
    """Run explain_evaluate and render Streamlit widgets; handles early exit string."""
    raw = exp_m.explain_evaluate(
        user_data=st.session_state["data"], verbose=False, return_all=True
    )
    if isinstance(raw, str):
        st.info(raw)
        return

    if prompt_type == "tot":
        out, explanation, code2, final_cf, code3, final_df, prediction, n_rules, rules_followed, first_rule, second_rule, third_rule, in_data = raw
        with st.expander("Combined output"):
            st.markdown(out)
        with st.expander("Explanation"):
            st.markdown(explanation)
        with st.expander("Evaluation"):
            st.subheader("Generating Example")
            st.code(code2, "python")
            st.dataframe(final_df)
            st.subheader("Evaluating example")
            st.code(code3, "python")
            st.text("Predicted class: " + str(prediction))
            st.text("Rules followed by the example: " + str(rules_followed))
            st.text("The example is in the original data: " + str(in_data))
            st.text("The example follows 1st ranked rule: " + str(first_rule))
            st.text("The example follows 2nd ranked rule: " + str(second_rule))
            st.text("The example follows 3rd ranked rule: " + str(third_rule))
    else:
        (
            counterfactuals,
            rules,
            code1,
            result1,
            explanation,
            code2,
            final_cf,
            code3,
            prediction,
            n_rules,
            rules_followed,
            first_rule,
            second_rule,
            third_rule,
            is_in_data,
        ) = raw
        with st.expander("Counterfactuals"):
            st.dataframe(counterfactuals)
        with st.expander("Identified Causes"):
            st.markdown(rules)
        with st.expander("Checking and Ranking Causes"):
            st.subheader("Code")
            st.code(code1, "python")
            st.subheader("Execution results")
            st.text(result1)
        with st.expander("Explanation"):
            st.markdown(explanation)
        with st.expander("Evaluation"):
            st.subheader("Generating Example")
            st.code(code2, "python")
            st.dataframe(final_cf)
            st.subheader("Evaluating example")
            st.code(code3, "python")
            st.text("Predicted class: " + str(prediction))
            st.text("Rules followed by the example: " + str(rules_followed))
            st.text("The example is in the original data: " + str(is_in_data))
            st.text("The example follows 1st ranked rule: " + str(first_rule))
            st.text("The example follows 2nd ranked rule: " + str(second_rule))
            st.text("The example follows 3rd ranked rule: " + str(third_rule))


def main():
    _dataset_options = (
        "Adult — income prediction",
        "German credit — credit risk",
        "Loan — loan approval",
    )
    dataset_label = st.sidebar.selectbox(
        "Dataset",
        _dataset_options,
        help="Switch between Adult census income, Statlog German Credit, and the loan approval task.",
    )
    if dataset_label == _dataset_options[0]:
        bundle_key = "adult"
    elif dataset_label == _dataset_options[1]:
        bundle_key = "german"
    else:
        bundle_key = "loan"

    if bundle_key == "adult":
        bundle = load_adult_bundle()
    elif bundle_key == "german":
        bundle = load_german_bundle()
    else:
        bundle = load_loan_bundle()

    if st.session_state.get("_demo_dataset") != bundle_key:
        st.session_state["_demo_dataset"] = bundle_key
        st.session_state["data"] = default_user_row(bundle)
        outcome = bundle["outcome_name"]
        st.session_state["cf_features_to_vary"] = [
            c for c in bundle["train_dataset"].columns if c != outcome
        ]

    if "cf_posthoc_sparsity_param" not in st.session_state:
        st.session_state["cf_posthoc_sparsity_param"] = 0.1
    if "cf_posthoc_sparsity_algorithm" not in st.session_state:
        st.session_state["cf_posthoc_sparsity_algorithm"] = "linear"
    if "cf_stopping_threshold" not in st.session_state:
        st.session_state["cf_stopping_threshold"] = 0.5
    if "cf_use_dice_seed" not in st.session_state:
        st.session_state["cf_use_dice_seed"] = False
    if "cf_dice_seed_value" not in st.session_state:
        st.session_state["cf_dice_seed_value"] = 42
    if "cf_use_sample_size" not in st.session_state:
        st.session_state["cf_use_sample_size"] = False
    if "cf_dice_sample_size" not in st.session_state:
        st.session_state["cf_dice_sample_size"] = 1000
    if "cf_permitted_range_json" not in st.session_state:
        st.session_state["cf_permitted_range_json"] = ""
    if "cf_permitted_range" not in st.session_state:
        st.session_state["cf_permitted_range"] = None

    if "parameters" not in st.session_state:
        st.session_state["parameters"] = "ok"
        st.session_state["prompt_type"] = "zero"
        st.session_state["n_cfs"] = 5
        st.session_state["llm"] = DEFAULT_LLM_MODEL
        st.session_state["llm_provider"] = DEFAULT_LLM_PROVIDER

    st.title(bundle["page_title"])

    with st.expander("User Data"):
        if bundle_key == "adult":
            with st.form("data_form"):
                age = st.text_input("Age", value=28)
                workclass = st.selectbox(
                    "Work class",
                    ("Private", "Self-Employed", "Other/Unknown", "Government"),
                    index=2,
                )
                education = st.selectbox(
                    "Education",
                    (
                        "Bachelors",
                        "Assoc",
                        "Some-college",
                        "School",
                        "HS-grad",
                        "Masters",
                        "Prof-school",
                        "Doctorate",
                    ),
                    index=1,
                )
                marital_status = st.selectbox(
                    "Marital status",
                    ("Single", "Married", "Divorced", "Widowed", "Separated"),
                    index=4,
                )
                occupation = st.selectbox(
                    "Occupation",
                    (
                        "White-Collar",
                        "Professional",
                        "Service",
                        "Blue-Collar",
                        "Other/Unknown",
                        "Sales",
                    ),
                    index=4,
                )
                race = st.selectbox("Race", ("White", "Other"), index=0)
                gender = st.selectbox("Gender", ("Female", "Male"))
                hours_per_week = st.text_input("Hours per week", value=40)

                submitted_data = st.form_submit_button(
                    "Save Data", use_container_width=True
                )
                if submitted_data:
                    st.session_state["data"] = pd.DataFrame(
                        {
                            "age": [int(age)],
                            "workclass": [workclass],
                            "education": [education],
                            "marital_status": [marital_status],
                            "occupation": [occupation],
                            "race": [race],
                            "gender": [gender],
                            "hours_per_week": [int(hours_per_week)],
                        }
                    )
        elif bundle_key == "german":
            train = bundle["train_dataset"]
            opts = bundle["german_cat_options"]
            cur = st.session_state["data"].iloc[0].to_dict()
            feature_order = [c for c in train.columns if c != bundle["outcome_name"]]

            with st.form("data_form_german"):
                values = {}
                for col in feature_order:
                    if col in GERMAN_CONTINUOUS_FEATURES:
                        lo = int(train[col].min())
                        hi = int(train[col].max())
                        default = int(cur[col])
                        values[col] = st.number_input(
                            col, min_value=lo, max_value=hi, value=default, key=f"g_{col}"
                        )
                    else:
                        choices = opts[col]
                        default_v = str(cur[col])
                        idx = choices.index(default_v) if default_v in choices else 0
                        values[col] = st.selectbox(
                            col, choices, index=idx, key=f"g_{col}"
                        )

                submitted_data = st.form_submit_button(
                    "Save Data", use_container_width=True
                )
                if submitted_data:
                    st.session_state["data"] = pd.DataFrame(
                        {c: [values[c]] for c in feature_order}
                    )
        else:
            train = bundle["train_dataset"]
            opts = bundle["loan_cat_options"]
            cur = st.session_state["data"].iloc[0].to_dict()
            feature_order = [c for c in train.columns if c != bundle["outcome_name"]]

            with st.form("data_form_loan"):
                values = {}
                for col in feature_order:
                    if col in LOAN_CONTINUOUS_FEATURES:
                        if col == "Dependents":
                            lo = int(train[col].min())
                            hi = int(train[col].max())
                            default = int(cur[col])
                            values[col] = st.number_input(
                                col, min_value=lo, max_value=hi, value=default, key=f"l_{col}"
                            )
                        else:
                            lo = float(train[col].min())
                            hi = float(train[col].max())
                            default = float(cur[col])
                            values[col] = st.number_input(
                                col,
                                min_value=lo,
                                max_value=hi,
                                value=default,
                                step=0.01 if col == "CoapplicantIncome" else 1.0,
                                key=f"l_{col}",
                            )
                    else:
                        choices = opts[col]
                        default_v = str(cur[col])
                        idx = choices.index(default_v) if default_v in choices else 0
                        values[col] = st.selectbox(
                            col, choices, index=idx, key=f"l_{col}"
                        )

                submitted_data = st.form_submit_button(
                    "Save Data", use_container_width=True
                )
                if submitted_data:
                    row = {}
                    for c in feature_order:
                        v = values[c]
                        if c in LOAN_CONTINUOUS_FEATURES:
                            if c == "Dependents":
                                row[c] = [int(v)]
                            else:
                                row[c] = [float(v)]
                        elif c == "Credit_History":
                            row[c] = [float(v)]
                        else:
                            row[c] = [v]
                    st.session_state["data"] = pd.DataFrame(row)

    with st.expander("Parameters"):
        with st.form("param_form"):
            st.markdown("**DiCE counterfactual search**")
            outcome = bundle["outcome_name"]
            feat_cols = [
                c for c in bundle["train_dataset"].columns if c != outcome
            ]
            prev_sel = st.session_state.get("cf_features_to_vary", feat_cols)
            prev_valid = [c for c in prev_sel if c in feat_cols] or feat_cols
            cf_feature_selection = st.multiselect(
                "Features DiCE may change (others stay fixed)",
                options=feat_cols,
                default=prev_valid,
                help="DiCE `features_to_vary`: only these columns are allowed to change when generating counterfactuals.",
            )
            cf_stopping = st.slider(
                "Stopping threshold (target-class probability)",
                min_value=0.05,
                max_value=1.0,
                value=float(st.session_state.get("cf_stopping_threshold", 0.5)),
                step=0.05,
                help="DiCE `stopping_threshold`: minimum predicted probability for the desired class on a candidate CF.",
            )
            cf_sparse = st.slider(
                "Post-hoc sparsity (continuous features)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("cf_posthoc_sparsity_param", 0.1)),
                step=0.05,
                help="DiCE `posthoc_sparsity_param`: higher values push continuous values closer to the original instance after generation.",
            )
            cf_sparse_algo = st.selectbox(
                "Post-hoc sparsity search",
                ("linear", "binary"),
                index=0
                if st.session_state.get("cf_posthoc_sparsity_algorithm", "linear")
                == "linear"
                else 1,
                help="DiCE `posthoc_sparsity_algorithm`: use binary when ranges are large and the outcome is monotonic in a feature.",
            )
            cf_perm_json = st.text_area(
                "Permitted value ranges (JSON, optional)",
                value=st.session_state.get("cf_permitted_range_json", ""),
                height=88,
                help='DiCE `permitted_range`, e.g. {"age": [20, 70], "Credit amount": [500, 10000]}. Empty = use training-derived ranges.',
            )
            use_seed = st.checkbox(
                "Set DiCE random seed (reproducible CFs)",
                value=st.session_state.get("cf_use_dice_seed", False),
            )
            seed_val = 42
            if use_seed:
                seed_val = st.number_input(
                    "Random seed",
                    min_value=0,
                    max_value=2**31 - 1,
                    value=int(st.session_state.get("cf_dice_seed_value", 42)),
                    step=1,
                )
            use_sample = st.checkbox(
                "Set DiCE sample_size (random method)",
                value=st.session_state.get("cf_use_sample_size", False),
                help="Optional `sample_size` for the internal sampling pool; leave off to use DiCE defaults.",
            )
            sample_sz = 1000
            if use_sample:
                sample_sz = st.number_input(
                    "sample_size",
                    min_value=1,
                    max_value=100_000,
                    value=int(st.session_state.get("cf_dice_sample_size", 1000)),
                    step=100,
                )

            st.markdown("**LLM**")
            llm_provider = st.radio(
                "LLM provider",
                options=("openai", "gemini"),
                format_func=lambda p: "OpenAI" if p == "openai" else "Google (Gemini)",
                horizontal=True,
                index=0
                if st.session_state.get("llm_provider", DEFAULT_LLM_PROVIDER)
                == "openai"
                else 1,
            )
            llm = st.text_input(
                "Model name",
                value=st.session_state.get("llm", DEFAULT_LLM_MODEL),
                help="OpenAI examples: gpt-4o, gpt-4, gpt-3.5-turbo. Google examples: gemini-3.1-flash-lite, gemini-2.0-flash, gemini-1.5-pro.",
            )
            prompt_type_label = st.selectbox(
                "Prompt", ("Zero Shot", "One Shot", "Tree of Thought (ToT)"), index=0
            )
            n_cfs = st.slider(
                "# of counterfactuals generated", min_value=1, max_value=5, value=5, step=2
            )

            submitted_parameters = st.form_submit_button(
                "Save Parameters", use_container_width=True
            )
            if submitted_parameters:
                st.session_state["parameters"] = "ok"
                if prompt_type_label == "Zero Shot":
                    st.session_state["prompt_type"] = "zero"
                elif prompt_type_label == "One Shot":
                    st.session_state["prompt_type"] = "one"
                else:
                    st.session_state["prompt_type"] = "tot"
                st.session_state["n_cfs"] = n_cfs
                st.session_state["llm"] = llm.strip() or DEFAULT_LLM_MODEL
                st.session_state["llm_provider"] = llm_provider

                st.session_state["cf_features_to_vary"] = (
                    cf_feature_selection if cf_feature_selection else list(feat_cols)
                )
                st.session_state["cf_stopping_threshold"] = float(cf_stopping)
                st.session_state["cf_posthoc_sparsity_param"] = float(cf_sparse)
                st.session_state["cf_posthoc_sparsity_algorithm"] = cf_sparse_algo
                st.session_state["cf_permitted_range_json"] = cf_perm_json
                pr = None
                if cf_perm_json.strip():
                    try:
                        pr = json.loads(cf_perm_json)
                        if not isinstance(pr, dict):
                            raise TypeError("permitted_range must be a JSON object")
                    except (json.JSONDecodeError, TypeError) as e:
                        st.error(f"Invalid permitted-range JSON: {e}")
                        pr = None
                st.session_state["cf_permitted_range"] = pr

                st.session_state["cf_use_dice_seed"] = use_seed
                st.session_state["cf_dice_seed_value"] = int(seed_val)
                st.session_state["cf_use_sample_size"] = use_sample
                st.session_state["cf_dice_sample_size"] = int(sample_sz)

    with st.form("explain_form"):
        submitted_explain = st.form_submit_button(
            "Explain Case", use_container_width=True
        )
        if submitted_explain:
            model = bundle["model"]
            train_dataset = bundle["train_dataset"]
            test_dataset = bundle["test_dataset"]

            st.subheader("Data")
            st.dataframe(st.session_state["data"])
            classif = int(pd.Series(model.predict(st.session_state["data"])).iloc[0])

            if bundle_key == "adult":
                if classif == 1:
                    st.markdown(
                        '<div class="green-box">'
                        + "This user is predicted to earn over 50k$ a year"
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="red-box">'
                        + "This user is predicted to earn less than 50k$ a year"
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            elif bundle_key == "german":
                if classif == 1:
                    st.markdown(
                        '<div class="green-box">'
                        + "This applicant is predicted to have good credit"
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="red-box">'
                        + "This applicant is predicted to have bad credit"
                        + "</div>",
                        unsafe_allow_html=True,
                    )
            else:
                if classif == 1:
                    st.markdown(
                        '<div class="green-box">'
                        + "This applicant is predicted to have the loan approved"
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="red-box">'
                        + "This applicant is predicted to have the loan denied"
                        + "</div>",
                        unsafe_allow_html=True,
                    )

            pt = st.session_state["prompt_type"]
            fv = st.session_state.get("cf_features_to_vary")
            outcome = bundle["outcome_name"]
            all_feats = [c for c in train_dataset.columns if c != outcome]
            if fv is not None and set(fv) == set(all_feats):
                features_to_vary = None
            else:
                features_to_vary = list(fv) if fv else None

            common_kwargs = dict(
                model=model,
                model_description=bundle["model_description"],
                backend="sklearn",
                dataset_info=bundle["dataset_info"],
                continuous_features=bundle["continuous_features"],
                outcome_name=outcome,
                training_set=train_dataset,
                test_set=test_dataset,
                llm=st.session_state.get("llm", DEFAULT_LLM_MODEL),
                llm_provider=st.session_state.get(
                    "llm_provider", DEFAULT_LLM_PROVIDER
                ),
                n_counterfactuals=st.session_state["n_cfs"],
                user_input=False,
                positive_prediction_message=bundle["positive_prediction_message"],
                features_to_vary=features_to_vary,
                permitted_range=st.session_state.get("cf_permitted_range"),
                stopping_threshold=float(
                    st.session_state.get("cf_stopping_threshold", 0.5)
                ),
                posthoc_sparsity_param=float(
                    st.session_state.get("cf_posthoc_sparsity_param", 0.1)
                ),
                posthoc_sparsity_algorithm=st.session_state.get(
                    "cf_posthoc_sparsity_algorithm", "linear"
                ),
                dice_random_seed=int(st.session_state["cf_dice_seed_value"])
                if st.session_state.get("cf_use_dice_seed")
                else None,
                dice_sample_size=int(st.session_state["cf_dice_sample_size"])
                if st.session_state.get("cf_use_sample_size")
                else None,
            )

            if pt == "tot":
                exp_m = ToTLLMExplanation4CFs(
                    prompt_type="zero",
                    branches=3,
                    **common_kwargs,
                )
                exp_m.fit()
                _render_explain_result(bundle, exp_m, "tot")
            else:
                exp_m = LLMExplanation4CFs(
                    prompt_type=st.session_state["prompt_type"],
                    **common_kwargs,
                )
                exp_m.fit()
                _render_explain_result(bundle, exp_m, st.session_state["prompt_type"])


main()
