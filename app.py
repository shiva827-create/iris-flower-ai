import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. AI మోడల్ ట్రైనింగ్
iris = load_iris()
model = RandomForestClassifier(random_state=42)
model.fit(iris.data, iris.target)

# వెబ్‌సైట్ పేజీ సెట్టింగ్స్
st.set_page_config(page_title="Iris AI Predictor", page_icon="🌸", layout="centered")

# 2. వెబ్‌సైట్ మెయిన్ హెడ్డింగ్
st.title("🌸 ఐరిస్ పూల గుర్తింపు AI వెబ్‌సైట్")
st.write("---")
st.markdown("### **ఎడమ వైపు ఉన్న స్లైడర్లలో కొలతలను మార్చి, పువ్వు రకాన్ని కనిపెట్టండి!**")

# 3. స్లైడర్లను సైడ్‌బార్ లోకి మార్చడం
st.sidebar.header("📏 పూల రెక్కల కొలతలు")
sepal_l = st.sidebar.slider("Sepal Length (పొడవు)", 4.0, 8.0, 5.8)
sepal_w = st.sidebar.slider("Sepal Width (వెడల్పు)", 2.0, 4.5, 3.0)
petal_l = st.sidebar.slider("Petal Length (పొడవు)", 1.0, 7.0, 4.0)
petal_w = st.sidebar.slider("Petal Width (వెడల్పు)", 0.1, 2.5, 1.3)

# పూల ఫోటోల లింక్స్ (ఇంటర్నెట్ నుండి నేరుగా లోడ్ అవుతాయి)
images = {
    'Setosa': "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
    'Versicolor': "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    'Virginica': "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# 4. బటన్ మరియు ప్రిడిక్షన్
if st.button("🌸 పువ్వు రకాన్ని కనిపెట్టు", use_container_width=True):
    features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    prediction = model.predict(features)
    
    names = ['Setosa (సెటోసా)', 'Versicolor (వర్సికలర్)', 'Virginica (వర్జినికా)']
    img_keys = ['Setosa', 'Versicolor', 'Virginica']
    
    selected_flower = names[prediction[0]]
    selected_img = images[img_keys[prediction[0]]]
    
    # ఆన్సర్ బాక్స్
    st.success(f"### 🎯 మన AI చెప్పిన సమాధానం: **{selected_flower}**")
    
    # ఫోటోను చూపించడం
    st.image(selected_img, caption=f"{selected_flower} పువ్వు", use_container_width=True)
