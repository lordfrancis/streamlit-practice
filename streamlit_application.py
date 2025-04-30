import streamlit as st
from PIL import Image
import pickle
import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()   
from xgboost import XGBClassifier

nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
        text=text.lower()
        y=[]
        #tokenization
        text=nltk.word_tokenize(text)
        for i in text:
            if i.isalnum():
                y.append(i)
        text=y[:]
        y.clear()
        #removing stopwords and punctuations
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
        text=y[:]
        y.clear()
    
        #stemming applied on text
        for i in text:
            y.append(ps.stem(i))
        return y




#remove SMS_Spam_Classifier name from path while deploying locally 

tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('mnb_spam_detector.pkl','rb'))

print(tfidf)
print(model)

st.title("SMS Spam classifier")

#content

st.image(Image.open('spam_image.jpeg'))

st.write("""
A spam classifier uses machine learning to distinguish between legitimate and unsolicited emails . it employs algorithm to analyze content and other features to flag emails spam or not spam.

Algorithm used to train the model is stacking classifier(SVM,NB,Xgboost)

"""
 
)
input_sms= st.text_area("Enter the message to check")


if st.button('Predict'):
    # 1. Preprocess
    tokens = transform_text(input_sms)
    
    # Join tokens back into a single document
    text_processed = " ".join(tokens)
    
    # 2. Vectorize - note we're passing a list with one string
    vector_input = tfidf.transform([text_processed]).toarray()
    print(type(vector_input))
    print(vector_input)
    vector_input = pd.DataFrame(vector_input, columns=tfidf.get_feature_names_out())

    # 3. Predict
    prediction = model.predict(vector_input)[0]
    
    # 4. Display
    if prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
        

# if st.button('Predict'):

#     #1.preprocess    
#     transform_sms=transform_text(input_sms) 
#     print(type(transform_sms)) 
#     transform_sms=np.array(transform_sms) #converting the list of string format to array of string format

#     #2.vectorize
#     vector_input=tfidf.transform(transform_sms.astype('str')).toarray() 
#     print(type(vector_input)) 
#     print(vector_input) 
#     vector_input =    pd.DataFrame(vector_input,columns=tfidf.get_feature_names_out())

#     #3.predict

#     prediction= model.predict(vector_input)[0]
#     #4.display
#     #st.header("Spam") if prediction else st.header("Not Spam")
#     if prediction==1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")
      