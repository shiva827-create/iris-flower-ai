import streamlit as st
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. AI మోడల్ ని ఇక్కడే రన్ చేస్తున్నాం
iris = load_iris()
model = RandomForestClassifier(random_state=42)
model.fit(iris.data, iris.target)

# 2. వెబ్‌సైట్ డిజైన్
st.title("🌸 ఐరిస్ పూల గుర్తింపు AI వెబ్‌సైట్")
st.write("కింద ఉన్న కొలతలను మార్చి, అది ఏ రకం పువ్వో కనిపెట్టండి!")

sepal_l = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_w = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_l = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_w = st.slider("Petal Width", 0.1, 2.5, 1.3)

if st.button("🌸 Puwvu Rakanni Kanipettu"):
    features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    prediction = model.predict(features)
    names = ['Setosa (సెటోసా)', 'Versicolor (వర్సిkలర్)', 'Virginica (వర్జినికా)']
    st.success(f"🎯 AI Answer: {names[prediction[0]]}")