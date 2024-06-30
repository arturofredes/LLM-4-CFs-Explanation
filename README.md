# LLMs for Exaplaining Sets of Counterfactual Examples to Final Users
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
You will neead an Open AI API key in your environment variables.
```
OPENAI_API_KEY = your_key
```
Install necessary packages
```
pip install -r requirements.txt
```
## Example Notebook
This notebook contains an example of a case that is predicted to earn less than 50k$ a year. In this notebook we go over all of the steps followed in order to generate an explanation and evaluating it.

## Train Notebook
In this notebook we train a model to predict whether a person will earn more or less than 50k$. In here we also generate the file swith the data that we will have to read in other notebooks. This data is already included in the `data/` folder and the model in `models/` so there it is not strictly necessary to run this notebook.

## Experiments Notebook
In here all the code and results for the experiments are included.