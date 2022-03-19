import json
import requests
import streamlit as st 


st.set_page_config(layout="centered", page_title="Pizza Requester", page_icon="🍕")
st.title("🍕 Pizza Requester")
st.subheader("Make your request and get your pizza delivered!")

input_text = st.text_area("Enter your request here:", "", height=250)
if st.button("Submit"):
    r = requests.post("http://api/predict", data=json.dumps({"sample": input_text.lower()})).json()
    if r["prediction"] == "True":
        st.success("🎉 Your pizza will be delivered! 🎉")
        st.write("Enjoy your 🍕.")
    else:
        st.error("Sorry, we could not deliver your pizza. 😔")
    st.write(r)
