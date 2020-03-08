import streamlit as st

import joblib
from PIL import Image

gender_vectorizer = open("models/gender_vectorizer.pkl", "rb")
gender_cv = joblib.load(gender_vectorizer)

gender_nv_model = open("models/naivebayesgendermodel.pkl", "rb")
gender_clf = joblib.load(gender_nv_model)


@st.cache
def predict_gender(name):
    vect = gender_cv.transform(name).toarray()
    result = gender_clf.predict(vect)
    return result

def load_images(image_name):
    img = Image.open(image_name)
    return st.image(img, width=300)

def load_css(css_file):
    with open(css_file) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
    
def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)

def main():
    st.title("Gender Classifier ML App")

    html_temp = """
    <div style="background-color:tomato;padding:15px">
    <h2> Streamlit ML APP</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    load_css('icon.css')
    load_icon('people')
    name = st.text_input("Enter the name:", "Type here")
    if st.button("Classify"):
        st.text("Name: {}".format(name.title()))
        result = predict_gender([name])
        if result[0] == 0:
            prediction = 'Female'
            c_image = 'female.png'
        else:
            prediction = 'Male'
            c_image = 'male.png'
        st.success("Name {}, was classified as {}.".format(name.title(), prediction))
        load_images(c_image)





if __name__ == "__main__":
    main()