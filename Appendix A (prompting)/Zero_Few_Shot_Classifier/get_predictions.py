import pandas as pd
from predicting_functions import *
import time 
#read the test data
#df=pd.read_csv('Data/test_data.csv')
df=pd.read_csv('./test_data_big.csv')
#time.sleep(7200)

#df=pd.read_csv('./tripadvisor_hotel_reviews.csv')

#predict_df(df,'Review', iterations = 3, shots = 'few',  model='gpt-4-1106-preview', explanation=True, confidence = True, role = 'expert hotelier' ,client=client)
#print('experiment 1 done')

#predict_df(df,'Review', iterations = 1, shots = 'zero',  model='gpt-3.5-turbo', explanation=False, confidence = True, role = 'expert hotelier' ,client=client)
#print('experiment 2 done')

#predict_df(df,'Review', iterations = 1, shots = 'few',  model='gpt-3.5-turbo', explanation=True, confidence = True, role = 'expert hotelier' ,client=client)
#print('experiment 3 done')

#predict_df(df,'Review', iterations = 2, shots = 'few',  model='gpt-3.5-turbo', explanation=True, confidence = True, role = 'expert hotelier' ,client=client)
print('experiment 4 done')

#predict_df(df,'Review', iterations = 3, shots = 'few',  model='gpt-3.5-turbo', explanation=True, confidence = True, role = 'expert hotelier' ,client=client)
#print('experiment 5 done')


"""We are trying to bypass the API limitation, to get results for the whole data set"""

iterations=1
for i in range(1,40):
    print(i*100)
    df_copy=df[i*100:(i+1)*100]
    done = False
    while done==False:
        try:
            # Apply the prompting function
            predictions = df_copy['Review'].apply(lambda x: prompting(x, iterations = 1, shots = 'few',  model='gpt-4-1106-preview', explanation=True, confidence = True, role = 'expert hotelier' ,client=client))  
            # Iterate through each prediction and apply extract_class_confidence
            for j in range(iterations):
                # Assuming extract_class_confidence returns a single value or a tuple, adapt accordingly
                df_copy['prediction_' + str(j)] = predictions.apply(lambda x: extract_from_output(x[j], 'class'))
                df_copy['confidence_'+ str(j)] = predictions.apply(lambda x: extract_from_output(x[j], 'confidence_level'))

            save_string = 'predictions_' +str((i+1)*100) + '.csv'
            print('Saved: ',save_string)
            df_copy.to_csv(save_string) 
            done = True
        except Exception as e:
            print('Error occurred:', e)
            time.sleep(3600*24)
