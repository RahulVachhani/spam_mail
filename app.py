import streamlit as st
import pickle

filename = 'spam_mail_model.sav'
load_model = pickle.load(open(filename, 'rb'))

feature_extraction = load_model['feature_extraction']
model = load_model['model']

def diabetes_prediction(input_data):
    s = feature_extraction.transform(input_data)
    prediction = model.predict(s)
    if (prediction[0] == 0):
      return 'The mail is spam mail'
    else:
      return 'The mail is not spam mail'
    


def main():
   
    st.title('Mail Prediction Web APP')


    message = st.text_area('Enter your mail')


    diagnosis = ''

    if st.button('Check'):
        diagnosis = diabetes_prediction([message])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
