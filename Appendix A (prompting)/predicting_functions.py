from openai import OpenAI
client = OpenAI()
def one_shot_predict(review, model='gpt-3.5-turbo', explanation=False, confidence = False, role = 'helpful assistant' ,client=client):

    if explanation == True:
        if confidence == True:
            conf_level = 'Confidence Level: <confidence_level>'
            extra_instr = 'giving a confidence level on the prediction from (low, medium, high)'
            conf_pos = '\nConfidence Level: high'
            conf_neg = '\nConfidence Level: high'
            conf_nt = '\nConfidence Level: medium'
        else:
            conf_level = ''
            extra_instr = ''
            conf_pos = ''
            conf_neg = ''
            conf_nt = ''
        instruction = """Explain the reasoning behind the classification briefly and classify the review {extra} as follows:
        Reasoning: <reasoning>
        \nClass: <class>
        \n{conf}
        """.format(extra = extra_instr, conf = conf_level)

        positive_explanation = """Reasoning: The reviewer highlights several positive aspects of their stay at Hotel Monaco. 
        They mention that the reception staff was friendly and professional, and they enjoyed the smart and comfortable room with a comfortable bed. 
        The reviewer also mentions that they particularly liked the fact that the reception staff was friendly towards a small dog and that the staff and 
        guests spoke and loved the dog. Although there is a mild negative aspect mentioned, which is the distance uphill to the local market and restaurants, 
        the overall tone of the review remains positive. The reviewer states that they had a great experience at the hotel, indicating that the positive elements outweigh the negative one.
        \nClass: positive""" + conf_pos


        negative_explanation = """"'Reasoning: The review highlights several negative aspects of the hotel experience.
        The customer encountered issues with their booking request for a nonsmoking room with a king bed away from the elevator and ice. 
        The front desk staff was rude and did not honor the special requests noted on the reservation card. They were given a room with two double beds located directly across from the elevator and ice machine. 
        The hotel parking garage was dirty, and there was a smell of urine in the hotel lobby. The carpets in the registration and elevator area were dirty, 
        and the upholstery and curtains in the room were also in need of cleaning. The bathroom had issues with water conservation, with the tub facet constantly dripping and the shower curtain having holes. 
        There was no coffee maker in the room, and the bedding did not fit the bed properly. The staff, except for the housekeeping staff, were described as unfriendly and short. 
        Overall, the review indicates a negative experience at the hotel.
        \nClass: negative'
        """ + conf_neg
        neutral_explanation = """"Reasoning: the review contains both positive and negative elements. 
        On one hand, it highlights the excellent location of the hotel, mentioning its proximity to Pike Market and great restaurants. 
        On the other hand, it criticizes the quality of the hotel, comparing it unfavorably to others and pointing out several issues. 
        This mix of positive and negative aspects can lead to a neutral overall sentiment.
        The language used in the review is relatively moderate. It doesn't use strong positive words (like "amazing" or "fantastic") or strongly negative words (like "horrible" or "terrible"). The reviewer states facts and observations without strong emotional emphasis.  
        The reviewer lists specific issues (like no desks in rooms, worn furniture coverings, and the type of air conditioning) but does so in a factual manner without exaggeration.
        \nClass: neutral
        """ + conf_nt
    else:
        if confidence == True:
            instruction = """Classify the review and give a confidence level on the prediction from (low, medium, high) as follows:
            Class: <class>
            \nConfidence Level: <confidence_level>
            """
            conf_pos = '\nConfidence Level: high'
            conf_neg = '\nConfidence Level: high'
            conf_nt = '\nConfidence Level: medium'
        else:
            instruction = "Do it as follows: Class: <class>"
            conf_pos = ''
            conf_neg = ''
            conf_nt = ''
        positive_explanation = 'Class: positive' + conf_pos
        negative_explanation = 'Class: negative' + conf_neg
        neutral_explanation = 'Class: neutral' + conf_nt

    good_review="""excellent stayed hotel monaco past w/e delight, reception staff friendly professional room smart comfortable bed, 
    particularly liked reception small dog received staff guests spoke loved, 
    mild negative distance uphill ppmarket restaurants 1st, overall great experience,  '"""

    bad_review = """bad choice, booked hotel hot wire called immediately requesting nonsmoking room king bed room away elevator/ice.
    the person spoke pleasant stated not guarantee requests honored make note reservation, check-in person desk rude said no special 
    request noted reservation card andstated no king beds way reservation stuck 2 double beds, located directly accross elevator ice, 
    nonsmoking, no elevator parking garage hotel, warwick mats garage filthy stairwells, hotel faces 4th smells urine, carpets registration elevator area need cleaning, 
    upholstery curtains room needed cleaning andpressing sign bathroom water conservation tub facet dripped continuously, tub drain needsattention shower curtain holes, 
    no coffee maker room, bedding did not fit bed sleeping directly mattress bedding askew, staff unfriendly short, 
    housekeeping staff quite pleasant, stay hotel, """

    neutral_review="""expensive, not biz travellers, simple fact hotel location simply unbeatable.. 
    mere stone throw away pike market, plenty great restaurants generally fun area, tourist, quality hotel so-so, 
    not place business travellers, bit hard especially just returned trip hk stayed conrad hk wynn macau, 
    compared inn feels like motel 6. issues:1. no desks rooms place laptops, wireless internet no place work bed,
      2. furniture coverings worned just outdated, 3. ac standalone unit attached wall, not central, 
    standalone unit quite bit noise, looks cheap tacky"""

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a" + role + " who has to label reviews as 'positive', 'neutral'  or 'negative'." + instruction,
            },
            
            {
                "role": "user",
                "content": good_review,
            },

            {
                "role": "assistant",
                "content": positive_explanation,
            },

            {
                "role": "user",
                "content": neutral_review,
            },

            {
                "role": "assistant",
                "content": neutral_explanation ,
            },    
            
            {
                "role": "user",
                "content": bad_review,
            },
            
            {
                "role": "assistant",
                "content": negative_explanation,
            },

            {
                "role": "user",
                "content": review,
            },
        ],
        model = model,
    )
    return chat_completion.choices[0].message.content

def zero_shot_predict(review,model='gpt-3.5-turbo',explanation=False, confidence = False, role = 'helpful assistant', client=client):
    if explanation == True:
        if confidence == True:
            conf_level = 'Confidence Level: <confidence_level>'
            extra_instr = 'giving a confidence level on the prediction from (low, medium, high)'
        else:
            conf_level = ''
            extra_instr = ''
        instruction = """Explain the reasoning behind the classification step by step briefly and classify the review {extra} as follows:
        Reasoning: <reasoning>
        \nClass: <class>
        \n{conf}
        """.format(extra = extra_instr, conf = conf_level)
    else:
        if confidence == True:
            instruction = """Classify the review and give a confidence level on the prediction from (low, medium, high) as follows:
            \nClass: <class>
            \nConfidence Level: <confidence_level>
            """
        else:
            instruction = "Do it as follows: Class: <class>"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are " + role  + " who has to label reviews as 'positive', 'neutral' or 'negative'. " + instruction,
            },
            
            {
                "role": "user",
                "content": review,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content

def get_embedding_gpt(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def prompting(review, iterations = 1, shots = 'zero',  model='gpt-3.5-turbo', explanation=False, confidence = False, role = 'helpful assistant' ,client=client ):
    results = []
    if shots == 'few':
        predicting_function = one_shot_predict
    elif shots == 'zero':
        predicting_function = zero_shot_predict
    else:
        print('strategy not valid')
        return None

    for i in range(iterations):
        output = predicting_function(review=review, model=model, explanation=explanation,confidence = confidence, role=role, client = client )
        results.append(output)
    return results

def extract_from_output(output, to_extract = 'class'):
    # Split the output into lines
    lines = output.split('\n')

    # Search for the line that contains 'Class'
    if to_extract == 'reasoning':
        line_start = 'Reasoning'
    elif to_extract == 'confidence_level':
        line_start = 'Confidence Level'
    elif to_extract == 'class':
        line_start = 'Class'
    else:
        print('Not a part of the output')
        return None
    for line in lines:
        if line.startswith(line_start):
            # Split the line into 'Class' and the actual class value
            _, value = line.split(': ')
            return value.strip().lower()

def predict_df(df, reviews_column, iterations = 1, shots = 'zero',  model='gpt-3.5-turbo', explanation=False, confidence = False, role = 'helpful assistant' ,client=client):
    df_copy=df.copy()
    # Apply the prompting function
    predictions = df_copy[reviews_column].apply(lambda x: prompting(x, iterations=iterations, shots=shots, model=model, explanation=explanation, confidence=confidence, role=role, client=client))
    if confidence == True:
        # Iterate through each prediction and apply extract_class_confidence
        for i in range(iterations):
            # Assuming extract_class_confidence returns a single value or a tuple, adapt accordingly
            df_copy['prediction_' + str(i)] = predictions.apply(lambda x: extract_from_output(x[i], 'class'))
            df_copy['confidence_'+ str(i)] = predictions.apply(lambda x: extract_from_output(x[i], 'confidence_level'))

    else:
        # Iterate through each prediction and apply extract_class_confidence
        for i in range(iterations):
            # Assuming extract_class_confidence returns a single value or a tuple, adapt accordingly
            df_copy['prediction_' + str(i)] = predictions.apply(lambda x: extract_from_output(x[i], 'class'))
    
    save_string = 'predictions_' + str(iterations) + '_' + str(shots) + '_' + str(model) + '_' + 'ChOT'+ '_' + str(explanation) + '_conf'+ '_' + str(confidence) + '_' + str(role) + '.csv'
    print('Saved: ',save_string)
    df_copy.to_csv(save_string)  