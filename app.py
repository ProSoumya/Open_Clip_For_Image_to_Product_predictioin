import streamlit as st
import torch
import cv2
import albumentations as A
import torch.nn.functional as F
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from model.clip_model import CLIPModel
st.title("Product Image to description Prediction in E-commerce")

def get_emebddings(file_path):
    with open(file_path,'rb') as file:
        data = pickle.load(file)

    return data


# def find_text_matches(model,text_emebddings,)


embeddings_data_path = Path("./data/embeddings.pkl")
image_caption_path = Path("./data/image_details.csv")
model_path = Path('./model/best.pt')
clip_model = CLIPModel().to('cpu')
clip_model.load_state_dict(torch.load(model_path,map_location='cpu'))
embeddings = get_emebddings(embeddings_data_path)
caption_df = pd.read_csv(image_caption_path)
# print(caption_df.head())



def find_text_matches(model, text_emebddings, image_path,actual_captions,max_out=4):
    item={}
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
                        A.Resize(224,224,always_apply=True),
                        A.Normalize(max_pixel_value=255.0,always_apply=True)
                            ])
    trans_image = transform(image=image)['image']
    item['image'] = torch.tensor(trans_image).permute(2,0,1).float().unsqueeze(0)
    
    #Prediction
    with torch.no_grad():
        image_features = model.image_encoder(item['image'].to('cpu'))
        image_embeddings = model.image_projection(image_features)
        image_embeddings_n = F.normalize(image_embeddings,p=2,dim=-1)
        text_embeddings_n = F.normalize(text_emebddings,p=2,dim=-1)
        dot_similarity = text_embeddings_n @ image_embeddings_n.T
        values,indices = torch.topk(dot_similarity.T.cpu() ,k=20)
        matches = [actual_captions[idx] for idx in indices[::5]]
    return matches



st.subheader("Select the Image from Given files path")
images = ("./images/0108775015.jpg","./images/0120129014.jpg","./images/0187949019.jpg","./images/0203595036.jpg","./images/0212629031.jpg","./images/0212629048.jpg","./images/0237347052.jpg")
image = st.selectbox("images",images)
st.subheader("Selected Image")
st.image(image)
ok = st.button("Predict")
if ok:
    # st.write("true")
    st.write("Predicted Product Description")
    matches = find_text_matches(clip_model,embeddings,image,caption_df['caption'].values)
    for i in matches:
        st.write(i)



