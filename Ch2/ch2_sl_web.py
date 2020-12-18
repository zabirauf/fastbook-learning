from fastbook import *

import streamlit as st

learn_inf = load_learner(Path('./export.pkl'))

uploaded_file = st.file_uploader("Choose a file")

if  uploaded_file is not None:
    read_image = uploaded_file.read()
    img = PILImage.create(read_image)
    st.image(img.to_thumb(688,688))
    
    #with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    st.text(f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}')
