
from keras.models import model_from_json
from keras.preprocessing import image
from PIL import Image
import numpy as np
import streamlit as st
from io import BytesIO, StringIO
#load model
model = model_from_json(open("model.json", "r").read())
#load weights
model.load_weights('model.h5')

def predict_class(image):
    results = {
        0: 'aeroplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    im = image.resize((32, 32))
    im = np.expand_dims(im, axis=0)
    im = np.array(im)
    pred = model.predict_classes([im])[0]
    # print(pred, results[pred])
    return results[pred]


def main():
    st.title("Cifar-10 Image Classification")
    html_temp = """
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;"> Cifar-10 Image Classification App </h2>
    </div>
    """
    #st.info(__doc__)
    st.markdown(html_temp, unsafe_allow_html=True)

    file = st.file_uploader("Upload your image", type="jpg")
    show_file = st.empty()
    if not file:
        show_file.info("Please upload a file of type: jpg")
        return
    content = file.getvalue()
    if isinstance(file, BytesIO):
        show_file.image(file)
    image = Image.open(file)
    result = ""
    if st.button("What do u think this is ?"):
        result = predict_class(image)
    st.success("Mmm I think it's a (an): {}".format(result))


if __name__ == '__main__':
    main()