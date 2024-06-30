################### Zero Shot Prompts #############################
def ZeroShotRules():
    return  """
        Im providing a negative outcome from a {ML-system} and your task is to extract the most important observed rules based on a set of counterfactual cases. 
        ----- Negative assessment outcome -----
        {negative_outcome}

        ----- Positive couterfactual outcome -----
        {positive_outcome}

        ----- Rules -----
        <List of Rules>
        """

def ZeroShotRulesCode():
    return  """
        Im providing a negative outcome from a {ML-system}, a set of counterfactual cases that flip the decision of the system and the main rules inferred from the counterfactuals.
        You should generate python code to count how many of the counterfactuals are consistent with the rule. The code should create a df with the counterfactuals provided and then check for each rule how many of them follow the rules. Order the rules. Finally, you should print the results.
 
        ----- Negative assessment outcome -----
        {negative_outcome}

        ----- Positive couterfactual outcome -----
        {positive_outcome}

        ----- Rules -----
        {rules}
        
        ----- Dataset info -----
        The following info about the dataset is available:
        {dataset_info}
        
        ----- Code -----
        ```
        import pandas as pd
        #complete this code
        ```
        """ 

def ZeroShotExplanation(user_input = False):
    if user_input:
        return  """
            A person has been classified in the negative class of {ML-system}. The data is the following:
            ----- Negative assessment outcome -----
            {negative_outcome}

            ----- Positive couterfactual outcome -----
            {positive_outcome}

            ----- Rules -----
            By generating counterfactuals, we obtained the following rules:
            {rules}


            ----- Results -----
            We have checked that the rules are followed by n counterfactuals:
            {results}

            ----- Dataset info -----
            The following info about the dataset is available:
            {dataset_info}

            ----- Explanation -----
            Given this information, provide an explanation to the user in plain language so that he/she can improve their chances of changing class. It should be as clear as possible and call to action. Consider that the higher amount of counterfactuals that follow the rule, the more important that rule is. Furthermore, an expert user has said that the most relevant rules are {user_input}
            <explanation>
            """
    else:
        return  """
            A person has been classified in the negative class of {ML-system}. The data is the following:
            ----- Negative assessment outcome -----
            {negative_outcome}

            ----- Positive couterfactual outcome -----
            {positive_outcome}

            ----- Rules -----
            By generating counterfactuals, we obtained the following rules:
            {rules}


            ----- Results -----
            We have checked that the rules are followed by n counterfactuals:
            {results}

            ----- Dataset info -----
            The following info about the dataset is available:
            {dataset_info}

            ----- Explanation -----
            Given this information, provide an explanation to the user in plain language so that he/she can improve their chances of changing class. It should be as clear as possible and call to action. Consider that the higher amount of counterfactuals that follow the rule, the more important that rule is. 
            <explanation>
            """

def ZeroShotExample():
    return """
        A person has been classified in the negative class of {ML-system}. The data is the following:
        ----- Negative assessment outcome -----
        {negative_outcome}

        
        ----- Explanation -----
        The following explanation was given inorder to try and change the class.
        {explanation}


        ----- Dataset info -----
        The following info about the dataset is available:
        {dataset_info}


        ----- Example -----
        Given this information, provide an example in the format of a pandas dataframe that would be in the positive class. Complete the code below and note that it is very important to use the name 'temp_csv.csv', since later processes rely on it.
        
        ```
        import pandas as pd
        df = pd.DataFrame(...) #complete this line
        df.to_csv('temp_csv.csv', index = False)

        ```
        """

def ZeroShotExampleCode():
    return  """
    Im providing a negative outcome from a {ML-system}. A counterfactual example in the format os a single row dataframe was created in temp_csv from the rules that are also provided. Give some code to check the number of rules followed by the example. The result must be given in the format of a dataframe and saved as a csv. The dataframe must have columns 'Rule' with the text of the rule, 'Importance' with the number of counterfactuals follow each rule, and 'In explanation' (1 or 0) depending if the final example follows the explanation or not. It is very important to save the csv as 'evaluation.csv'.
    
    ----- Negative assessment outcome -----
    {negative_outcome}

    ----- Rules -----
    {rules}

    ----- Results -----
    We have checked that the rules are followed by n counterfactuals:
    {results}
    
    ----- Dataset info -----
    The following info about the dataset is available:
    {dataset_info}

    ----- Code -----
    ```
    import pandas as pd
    df = pd.read_csv('temp_csv.csv')

    #COMPLETE CODE

    # Save to csv
    df_final.to_csv('evaluation.csv', index = False)
    ```
    """ 

################### One Shot Prompts #############################
def OneShotRules():
    return """
    Im providing a negative outcome from a {ML-system} and your task is to extract the most important observed rules based on a set of counterfactual cases. 
    Example:
    ----- Negative assessment outcome -----
    (<bound method DataFrame.to_string of    age workclass education marital_status   occupation   race  gender  \
    0   29   Private   HS-grad        Married  Blue-Collar  White  Female   

    hours_per_week  income  
    0              38       0  >,)
    ----- Positive couterfactual outcome -----
    <bound method DataFrame.to_string of    age   workclass    education marital_status     occupation   race  gender  \
    0   29  Government        Assoc        Married    Blue-Collar  White  Female   
    1   29     Private        Assoc        Married        Service  White  Female   
    2   29     Private  Prof-school        Married   Professional  White  Female   
    3   29     Private      Masters        Married  Other/Unknown  White  Female   
    4   29     Private    Doctorate        Married    Blue-Collar  White    Male   

    hours_per_week  income  
    0              38       1  
    1              38       1  
    2              38       1  
    3              38       1  
    4              38       1  >
    ----- Rules -----
    1. Higher education (Masters, Doctorate, Bachelors, Prof-school) leads to higher income.
    2. Sales, Professional, and White-Collar occupations lead to higher income.
    3. Gender being Male leads to higher income.
    

    Problem:
    ----- Negative assessment outcome -----
    {negative_outcome}
    ----- Positive couterfactual outcome -----
    {positive_outcome}
    ----- Rules -----
    <List of Rules>

    '''
    """

def OneShotRulesCode():

    return """
    Im providing a negative outcome from a {ML-system} and aset of rules extracted from counterfactuals, you should generate python code to count how many of the counterfactuals are consistent with the rule. The code should create a df with the counterfactuals provided and then check for each rule how many of them follow the rules. Order the rules. Finally, you should print the results.
    Example:
    ----- Negative assessment outcome -----
    (<bound method DataFrame.to_string of    age workclass education marital_status   occupation   race  gender  \
    0   29   Private   HS-grad        Married  Blue-Collar  White  Female   

    hours_per_week  income  
    0              38       0  >,)
    ----- Positive couterfactual outcome -----
    <bound method DataFrame.to_string of    age   workclass    education marital_status     occupation   race  gender  \
    0   29  Government        Assoc        Married    Blue-Collar  White  Female   
    1   29     Private        Assoc        Married        Service  White  Female   
    2   29     Private  Prof-school        Married   Professional  White  Female   
    3   29     Private      Masters        Married  Other/Unknown  White  Female   
    4   29     Private    Doctorate        Married    Blue-Collar  White    Male   

    hours_per_week  income  
    0              38       1  
    1              38       1  
    2              38       1  
    3              38       1  
    4              38       1  >
    ----- Rules -----
    1. Higher education (Masters, Doctorate, Bachelors, Prof-school) leads to higher income.
    2. Sales, Professional, and White-Collar occupations lead to higher income.
    3. Gender being Male leads to higher income.

    ----- Dataset info -----
    The following info about the dataset is available:
    age: age
    workclass: type of industry (Government, Other/Unknown, Private, Self-Employed)
    education: education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)
    marital_status: marital status (Divorced, Married, Separated, Single, Widowed)
    occupation: occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)
    race: white or other race?
    gender: male or female?
    hours_per_week: total work hours per week
    income: 0 (<=50K) vs 1 (>50K)

    ----- Code -----
    '''
    import pandas as pd

    # Original negative outcome
    original = pd.DataFrame({{
        'age': [29],
        'workclass': ['Private'],
        'education': ['HS-grad'],
        'marital_status': ['Married'],
        'occupation': ['Blue-Collar'],
        'race': ['White'],
        'gender': ['Female'],
        'hours_per_week': [38],
        'income': [0]
    }})

    # Counterfactuals
    counterfactuals = pd.DataFrame({{
        'age': [29, 29, 29, 29, 29],
        'workclass': ['Private', 'Private', 'Private', 'Private', 'Private'],
        'education': ['Masters', 'Doctorate', 'Masters', 'Bachelors', 'Prof-school'],
        'marital_status': ['Married', 'Married', 'Married', 'Married', 'Married'],
        'occupation': ['Sales', 'Blue-Collar', 'Professional', 'Service', 'White-Collar'],
        'race': ['White', 'White', 'White', 'White', 'White'],
        'gender': ['Female', 'Male', 'Female', 'Female', 'Female'],
        'hours_per_week': [38, 38, 38, 38, 38],
        'income': [1, 1, 1, 1, 1]
    }})

    # Rule 1: Higher education leads to higher income
    rule1 = counterfactuals['education'].isin(['Masters', 'Doctorate', 'Bachelors', 'Prof-school']).sum()

    # Rule 2: Sales, Professional, and White-Collar occupations lead to higher income
    rule2 = counterfactuals['occupation'].isin(['Sales', 'Professional', 'White-Collar']).sum()

    # Rule 3: Gender being Male leads to higher income
    rule3 = (counterfactuals['gender'] == 'Male').sum()

    # Print results
    print("Number of counterfactuals following Rule 1:", rule1)
    print("Number of counterfactuals following Rule 2:", rule2)
    print("Number of counterfactuals following Rule 3:", rule3)
    '''

    Problem:
    ----- Negative assessment outcome -----
    {negative_outcome}

    ----- Positive couterfactual outcome -----
    {positive_outcome}

    ----- Rules -----
    {rules}

    ----- Dataset info -----
    The following info about the dataset is available:
    {dataset_info}
    
    ----- Code -----
    '''
    import pandas as pd
    <Code>
    '''
    """

def OneShotExplanation(user_input = False):
    if user_input:
        return """
        Example:
        Im providing a negative outcome from a {ML-system} and your task is to create an explanation for the end user leveraging all the information.
        ----- Negative assessment outcome -----
        (<bound method DataFrame.to_string of    age workclass education marital_status   occupation   race  gender  \
        0   29   Private   HS-grad        Married  Blue-Collar  White  Female   

        hours_per_week  income  
        0              38       0  >,)
        ----- Positive couterfactual outcome -----
        <bound method DataFrame.to_string of    age   workclass    education marital_status     occupation   race  gender  \
        0   29  Government        Assoc        Married    Blue-Collar  White  Female   
        1   29     Private        Assoc        Married        Service  White  Female   
        2   29     Private  Prof-school        Married   Professional  White  Female   
        3   29     Private      Masters        Married  Other/Unknown  White  Female   
        4   29     Private    Doctorate        Married    Blue-Collar  White    Male   

        hours_per_week  income  
        0              38       1  
        1              38       1  
        2              38       1  
        3              38       1  
        4              38       1  >
        ----- Rules -----
        1. Higher education (Masters, Doctorate, Bachelors, Prof-school) leads to higher income.
        2. Sales, Professional, and White-Collar occupations lead to higher income.
        3. Gender being Male leads to higher income.
        
        ----- Dataset info -----
        The following info about the dataset is available:
        age: age
        workclass: type of industry (Government, Other/Unknown, Private, Self-Employed)
        education: education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)
        marital_status: marital status (Divorced, Married, Separated, Single, Widowed)
        occupation: occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)
        race: white or other race?
        gender: male or female?
        hours_per_week: total work hours per week
        income: 0 (<=50K) vs 1 (>50K)

        ----- Results -----
        We have checked that the rules are followed by n counterfactuals:
        Number of counterfactuals following Rule 1: 5
        Number of counterfactuals following Rule 2: 3
        Number of counterfactuals following Rule 3: 1


        ----- Explanation -----
        Furthermore, an expert user has said that the most relevant rules are: 1,2
        Given this information, provide an explanation to the user in plain language so that he/she can improve their chances of changing class. It should be as clear as possible and call to action. Consider that the higher amount of counterfactuals that follow the rule, the more important that rule is. 
        
        Based on the results obtained from the analysis, it's clear that certain factors significantly influence the likelihood of achieving a higher income. Here's a plain language explanation targeting improvement:

        Pursue Higher Education: The analysis shows that obtaining a higher education degree such as a Masters, Doctorate, Bachelors, or Professional school significantly increases your chances of a higher income. This indicates that investing time and resources in further education can be a valuable path to improving your economic standing.

        Consider Your Occupation: Occupations categorized under Sales, Professional, and White-Collar are more likely to lead to higher incomes. If you're considering a career change or are at the start of your career, aiming for roles within these categories could improve your income prospects.

        Gender Influence: Although less impactful than education and occupation, the analysis suggests that gender plays a role, with males showing a higher likelihood of higher income in the given data set. While changing one's gender isn't a practical piece of advice, it highlights the importance of recognizing and, where possible, addressing gender disparities in the workplace.

        Improving your chances of moving to a higher income class involves strategic decisions about your education and career path. Focusing on acquiring higher education and targeting specific occupations can make a substantial difference. While the influence of gender is more complex and systemic, awareness and advocacy for equity in pay and opportunities remain essential.
        

        Problem:
        A person has been classified in the negative class of {ML-system}. The data is the following:
        ----- Negative assessment outcome -----
        {negative_outcome}


        ----- Rules -----
        By generating counterfactuals, we obtained the following rules:
        {rules}

        ----- Dataset info -----
        The following info about the dataset is available:
        {dataset_info}

        ----- Results -----
        We have checked that the rules are followed by n counterfactuals:
        {results}


        ----- Explanation -----
        Furthermore, an expert user has said that the most relevant rules are: {user_input}
        Given this information, provide an explanation to the user in plain language so that he/she can improve their chances of changing class. It should be as clear as possible and call to action. Consider that the higher amount of counterfactuals that follow the rule, the more important that rule is. 
        <explanation>
        """
    else:
        return """
        Im providing a negative outcome from a {ML-system} and your task is to create an explanation for the end user leveraging all the information provided. 
        Example:
        ----- Negative assessment outcome -----
        (<bound method DataFrame.to_string of    age workclass education marital_status   occupation   race  gender  \
        0   29   Private   HS-grad        Married  Blue-Collar  White  Female   

        hours_per_week  income  
        0              38       0  >,)
        ----- Positive couterfactual outcome -----
        <bound method DataFrame.to_string of    age   workclass    education marital_status     occupation   race  gender  \
        0   29  Government        Assoc        Married    Blue-Collar  White  Female   
        1   29     Private        Assoc        Married        Service  White  Female   
        2   29     Private  Prof-school        Married   Professional  White  Female   
        3   29     Private      Masters        Married  Other/Unknown  White  Female   
        4   29     Private    Doctorate        Married    Blue-Collar  White    Male   

        hours_per_week  income  
        0              38       1  
        1              38       1  
        2              38       1  
        3              38       1  
        4              38       1  >
        ----- Rules -----
        1. Higher education (Masters, Doctorate, Bachelors, Prof-school) leads to higher income.
        2. Sales, Professional, and White-Collar occupations lead to higher income.
        3. Gender being Male leads to higher income.
        
        ----- Dataset info -----
        The following info about the dataset is available:
        age: age
        workclass: type of industry (Government, Other/Unknown, Private, Self-Employed)
        education: education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)
        marital_status: marital status (Divorced, Married, Separated, Single, Widowed)
        occupation: occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)
        race: white or other race?
        gender: male or female?
        hours_per_week: total work hours per week
        income: 0 (<=50K) vs 1 (>50K)

        ----- Results -----
        We have checked that the rules are followed by n counterfactuals:
        Number of counterfactuals following Rule 1: 5
        Number of counterfactuals following Rule 2: 3
        Number of counterfactuals following Rule 3: 1


        ----- Explanation -----
        Given this information, provide an explanation to the user in plain language so that he/she can improve their chances of changing class. It should be as clear as possible and call to action. Consider that the higher amount of counterfactuals that follow the rule, the more important that rule is. 
        Based on the results obtained from the analysis, it's clear that certain factors significantly influence the likelihood of achieving a higher income. Here's a plain language explanation targeting improvement:

        Pursue Higher Education: The analysis shows that obtaining a higher education degree such as a Masters, Doctorate, Bachelors, or Professional school significantly increases your chances of a higher income. This indicates that investing time and resources in further education can be a valuable path to improving your economic standing.

        Consider Your Occupation: Occupations categorized under Sales, Professional, and White-Collar are more likely to lead to higher incomes. If you're considering a career change or are at the start of your career, aiming for roles within these categories could improve your income prospects.

        Gender Influence: Although less impactful than education and occupation, the analysis suggests that gender plays a role, with males showing a higher likelihood of higher income in the given data set. While changing one's gender isn't a practical piece of advice, it highlights the importance of recognizing and, where possible, addressing gender disparities in the workplace.

        Improving your chances of moving to a higher income class involves strategic decisions about your education and career path. Focusing on acquiring higher education and targeting specific occupations can make a substantial difference. While the influence of gender is more complex and systemic, awareness and advocacy for equity in pay and opportunities remain essential.
        

        Problem:
        A person has been classified in the negative class of {ML-system}. The data is the following:
        ----- Negative assessment outcome -----
        {negative_outcome}


        ----- Rules -----
        By generating counterfactuals, we obtained the following rules:
        {rules}

        ----- Dataset info -----
        The following info about the dataset is available:
        {dataset_info}

        ----- Results -----
        We have checked that the rules are followed by n counterfactuals:
        {results}


        ----- Explanation -----
        Given this information, provide an explanation to the user in plain language so that he/she can improve their chances of changing class. It should be as clear as possible and call to action. Consider that the higher amount of counterfactuals that follow the rule, the more important that rule is. 
        <explanation>
        """

def OneShotExample():
        return """
        Im providing a negative outcome from a {ML-system}, an explanation of why the outcome was negative, and some information about the dataset and variables.
        Given this information, your task is to generate a counterfactual example that will produce the desired output from the classifier. Provide the example in the format of a pandas dataframe that would be in the positive class. Complete the code below and note that it is very important to use the name 'temp_csv.csv', since later processes rely on it.
        Example:
        ----- Negative assessment outcome -----
        (<bound method DataFrame.to_string of    age workclass education marital_status   occupation   race  gender  \
        0   29   Private   HS-grad        Married  Blue-Collar  White  Female   

        hours_per_week  income  
        0              38       0  >,)
        
        ----- Explanation -----
        The following explanation was given inorder to try and change the class.
        According to the analysis of your data, there are a few steps you can take to increase your chances of being classified into the group that earns more than $50k a year.

        1. Improve your education: The most prominent rule derived from the counterfactuals is that having a higher level of education, such as a Bachelor's, Master's, or Professional-school degree, significantly increases your chances of earning more than $50k a year. So, if possible, consider furthering your education.

        2. Change your job sector/role: If you currently work in a blue-collar job, consider transitioning to a white-collar role or a role that falls under the "other/unknown" category. Additionally, working for the Government or in the Private sector also seems to improve your income.

        3. Gender considerations: While the system does indicate a positive outcome when the gender is male, it is essential to remember that this is a systemic issue and not something that you can or should change about yourself. 

        Remember, factors like your marital status, race, age, and the number of hours you work per week did not seem to have a significant impact on the income prediction in this model. So, the focus should be on improving your education and considering a job sector/role change.


        ----- Dataset info -----
        The following info about the dataset is available:
        age: age
        workclass: type of industry (Government, Other/Unknown, Private, Self-Employed)
        education: education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)
        marital_status: marital status (Divorced, Married, Separated, Single, Widowed)
        occupation: occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)
        race: white or other race?
        gender: male or female?
        hours_per_week: total work hours per week
        income: 0 (<=50K) vs 1 (>50K)


        ----- Example -----
        Given this information, provide an example in the format of a pandas dataframe that would be in the positive class. Complete the code below and note that it is very important to use the name 'temp_csv.csv', since later processes rely on it.

        
        ```python
        import pandas as pd

        # Define the data for the DataFrame
        data = {{
            'age': [29],
            'workclass': ['Government'],
            'education': ['Masters'],
            'marital_status': ['Married'],
            'occupation': ['Professional'],
            'race': ['White'],
            'gender': ['Female'],
            'hours_per_week': [38],
            'income': [1]
        }}

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Save to csv
        df.to_csv('temp_csv.csv', index = False)

        ```
            

        Problem:
        A person has been classified in the negative class of {ML-system}. The data is the following:
        ----- Negative assessment outcome -----
        {negative_outcome}

        ----- Explanation -----
        The following explanation was given inorder to try and change the class.
        {explanation}


        ----- Dataset info -----
        The following info about the dataset is available:
        {dataset_info}

        ----- Example -----
        Given this information, provide an example in the format of a pandas dataframe that would be in the positive class. Write the code 
        
        ```python
        import pandas as pd
        df = pd.DataFrame(...) #complete this line
        df.to_csv('temp_csv.csv', index = False)

        ```
        """

def OneShotExampleCode():

    return  """
    Im providing a negative outcome from a {ML-system}. A positive example in the format os a single row dataframe was created in temp_csv from the rules that are also provided. Give some code to check the number of rules followed by the example. The result must be given in the format of a dataframe and saved as a csv. The dataframe must have columns 'Rule' with the text of the rule, 'Importance' with the number of counterfactuals follow each rule, and 'In explanation' (1 or 0) depending if the final example follows the explanation or not. It is very important to save the csv as 'evaluation.csv'.
    Example:

    ----- Negative assessment outcome -----
    (<bound method DataFrame.to_string of    age workclass education marital_status   occupation   race  gender  \
    0   29   Private   HS-grad        Married  Blue-Collar  White  Female   

    hours_per_week  income  
    0              38       0  >,)

    ----- Rules -----
    1. Higher education (Masters, Doctorate, Bachelors, Prof-school) leads to higher income.
    2. Sales, Professional, and White-Collar occupations lead to higher income.
    3. Gender being Male leads to higher income.


    ----- Results -----
    We have checked that the rules are followed by n counterfactuals:
    Number of counterfactuals following Rule 1: 5
    Number of counterfactuals following Rule 2: 3
    Number of counterfactuals following Rule 3: 1     

    ----- Dataset info -----
    The following info about the dataset is available:
    age: age
    workclass: type of industry (Government, Other/Unknown, Private, Self-Employed)
    education: education level (Assoc, Bachelors, Doctorate, HS-grad, Masters, Prof-school, School, Some-college)
    marital_status: marital status (Divorced, Married, Separated, Single, Widowed)
    occupation: occupation (Blue-Collar, Other/Unknown, Professional, Sales, Service, White-Collar)
    race: white or other race?
    gender: male or female?
    hours_per_week: total work hours per week
    income: 0 (<=50K) vs 1 (>50K)

    ----- Code -----

    ```python
    #Read example
    import pandas as pd
    df = pd.read_csv('temp_csv.csv')


    # Define the data for the DataFrame
    data = {{
        'Rule': ['Higher education (Masters, Doctorate, Bachelors, Prof-school) leads to higher income.', 'Sales, Professional, and White-Collar occupations lead to higher income.', 'Gender being Male leads to higher income.']
        'Importance': ['5,3,1],
        'In explanation': [0,0,0]
    }}

    # Create the DataFrame
    df_final = pd.DataFrame(data)

    if df['workclass'].iloc[0] in ['Masters', 'Doctorate', 'Bachelors', 'Prof-school']:
        df_final['In explanation].iloc[0] = 1
    if df['education'].iloc[0] in ['Sales', 'Professional', 'White-Collar']:
        df_final['In explanation].iloc[1] = 1
    if df['gender'].iloc[0] == 'Male':
        df_final['In explanation].iloc[2] = 1

    # Save to csv
    df_final.to_csv('evaluation.csv', index = False)

    ```

    Problem:

    ----- Negative assessment outcome -----
    {negative_outcome}

    ----- Rules -----
    {rules}
    
    ----- Results -----
    We have checked that the rules are followed by n counterfactuals:
    {results}

    ----- Dataset info -----
    The following info about the dataset is available:
    {dataset_info}

    ----- Code -----
    ```python
    import pandas as pd
    df = pd.read_csv('temp_csv.csv')

    #Complete the code

    # Save to csv
    df_final.to_csv('evaluation.csv', index = False)
    ```
    
    
    """ 


################### ToTPrompts #############################
def ToT_explain():
    return """
    A negative outcome from a {ML-system} was provided to several systems that explain why that case is negative analyzing counterfactuals, generating rules and evaluating them.
    The results of the system are the following.

    {output_list} 

    Given this information, provide an explanation to the user in plain language so that he/she can improve their chances of changing class. It should be as clear as possible and call to action. Consider that the higher amount of counterfactuals that follow the rule, the more important that rule is. 


    """

def ToT_example():
    return """
    A negative outcome from a {ML-system} was provided to several systems that explain why that case is negative analyzing counterfactuals, generating rules and creating a final example that changes of class.
    The results of the system are the following.
    ----- Negative assessment outcome -----
    {negative_outcome}

    ----- Results of the systems -----
    {output_list} 
        
    
    ----- Dataset info -----
    The following info about the dataset is available:
    {dataset_info}

    
    Given this information, your task is to generate a counterfactual example that will be of the positive class. Provide the example in the format of a pandas dataframe that would be in the positive class. Complete the code below and note that it is very important to use the name 'temp_csv.csv', since later processes rely on it.
    ----- Code -----
    ```python
    import pandas as pd
    df = pd.DataFrame(...) #complete this line
    df.to_csv('temp_csv.csv', index = False)

    ```

    """

def ToT_ExampleCode():
    return  """
    Im providing a negative outcome from a {ML-system}. A positive example in the format os a single row dataframe was created in temp_csv from the rules that are also provided. Give some code to check the number of rules followed by the example. The result must be given in the format of a dataframe and saved as a csv. The dataframe must have columns 'Rule' with the text of the rule, 'Importance' with the number of counterfactuals follow each rule, and 'In explanation' (1 or 0) depending if the final example follows the explanation or not. You may group rules from different systems that are the same. It is very important to save the csv as 'evaluation.csv'.
    
    ----- Negative assessment outcome -----
    {negative_outcome}

    ----- Rules -----
    A negative outcome from a {ML-system} was provided to several systems that explain why that case is negative analyzing counterfactuals, generating rules and creating a final example that changes of class.
    The results of the system are the following.

    {output_list} 
    
    ----- Dataset info -----
    The following info about the dataset is available:
    {dataset_info}

    ----- Code -----
    ```
    import pandas as pd
    df = pd.read_csv('temp_csv.csv')

    #COMPLETE CODE

    # Save to csv
    df_final.to_csv('evaluation.csv', index = False)
    ```
    """ 