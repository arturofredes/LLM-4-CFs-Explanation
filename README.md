# LLMs for Exaplaining Sets of Counterfactual Examples to Final Users
- **📄 Paper:** [Using LLMs for Explaining Sets of Counterfactual Examples to Final Users](https://arxiv.org/abs/2408.15133)
- **📢 Presented at:** [2nd Workshop on Causal Inference and Machine Learning in Practice (@ KDD 2024)](https://causal-machine-learning.github.io/kdd2024-workshop/)
### Arturo Fredes & Jordi Vitrià 
Counterfactual examples have shown to be a promising method for explaining a ma-
chine learning model’s decisions, by providing the user with variants of its own data
with small shifts to flip the outcome. When a user is presented with a single coun-
terfactual, extracting conclusions from it is straightforward. Yet, this may not reflect
the whole scope of possible actions the user can take, and furthermore, the example
could be unfeasible. On the other hand, as we increase the number of counterfactu-
als, drawing conclusions from them becomes difficult for people who are not trained
in data analytic thinking. The objective of this work is to evaluate the use of LLMs
in producing clear explanations in plain language of these counterfactual examples
for the end user. We propose a method to decompose the explanation generation
problem into smaller, more manageable tasks to guide the LLM, drawing inspira-
tion from studies on how humans create and communicate explanations. We carry
out different experiments using a public dataset and propose a method of closed
loop evaluation to assess the coherence of the final explanation with the counterfac-
tuals as well as the quality of the content. Furthermore, an experiment with people
is currently being done in order to evaluate the understanding and satisfaction of
the users.

## Setup

### API keys

The Streamlit demo can call **OpenAI** or **Google Gemini** (set the provider in the app). Put keys in a `.env` file in the project root, or export them in your shell:

```
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_ai_studio_key
```

For Gemini-only use, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) is enough. For OpenAI, set `OPENAI_API_KEY`.

### Local install (without Docker)

```
pip install -r requirements.txt
streamlit run demo.py
```

### Run with Docker

The repo includes a `Dockerfile` and `docker-compose.yml` that build a small image and run the Streamlit app on port **8501**.

1. **Prepare files**  
   Ensure `data/` and `models/` are present (trained pickles such as `models/loan_model.pkl` and `models/german_credit_model.pkl`, plus the CSVs under `data/`). The image copies these into the container at build time.

2. **Create `.env`** in the same directory as `docker-compose.yml`, with at least the key for the LLM provider you will use (see API keys above). Compose loads this file automatically.

3. **Build and start** from the repository root:

   ```bash
   docker compose up --build
   ```

4. **Open the app** in a browser: [http://localhost:8501](http://localhost:8501).

5. **Stop** the stack: press `Ctrl+C` in the terminal, or run `docker compose down` from another shell in the same directory.

If port 8501 is already in use, change the host mapping in `docker-compose.yml` (for example `"8502:8501"`) and open `http://localhost:8502` instead.

To build and run without Compose (after creating `.env` or passing `-e` flags):

```bash
docker build -t llm-cfs-explanation-demo .
docker run --rm -p 8501:8501 --env-file .env llm-cfs-explanation-demo
```

## Example Notebook
This notebook contains an example of a case that is predicted to earn less than 50k$ a year. In this notebook we go over all of the steps followed in order to generate an explanation and evaluating it.

## Train Notebook
In this notebook we train a model to predict whether a person will earn more or less than 50k$. In here we also generate the file with the data that we will have to read in other notebooks. This data is already included in the `data/` folder and the model in `models/` so it is not strictly necessary to run this notebook.

## Experiments Notebook
In here all the code and results for the experiments are included.

## German Credit Notebook
In this notebook we preprocessed data of the german credit dataset, trained a classifier and generated explanations from some selected cases that will be shown to subjects.
