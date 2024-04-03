import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import boto3
from botocore.exceptions import NoCredentialsError
from tensorflow.keras.models import load_model

@st.cache_resource
# def load_model_from_s3(bucket_name, model_key):
#     import boto3
#     from botocore.exceptions import NoCredentialsError
#     import tensorflow as tf

#     s3 = boto3.client('s3')
#     local_model_path = 'Streamlit/local_CNN_model.h5'
#     try:
#         s3.download_file(bucket_name, model_key, local_model_path)
#         model = tf.keras.models.load_model(local_model_path)
#         return model
#     except NoCredentialsError:
#         st.error("AWS credentials not available. Check your configuration.")
#         return None


def predict(_image, _model):
    # Convert image to RGB
    if _image.mode != 'RGB':
        _image = _image.convert('RGB')

    test_image = tf.keras.preprocessing.image.img_to_array(_image.resize((224, 224)))
    test_image = preprocess_input(test_image)  # Ensure this matches your model's preprocessing
    test_image = np.expand_dims(test_image, axis=0)
    
    scores = _model.predict(test_image)
    class_names = ["Duck", "Kingfisher", "Pheasant", "Warbler"]  
    result = f"The uploaded image belongs to the genus {class_names[np.argmax(scores)]} "
    
    return result

def main():
    st.title('Bird Genus Classifier')
    st.subheader("Upload an image of a bird from the genus Duck, Kingfisher, Pheasant or Warbler, and the classifier will predict which genus it belongs to.")
    st.markdown("Please wait for the model to load during the initial setup, which should complete in about one minute.")
    # model = load_model_from_s3('larissahuang-bstn-bucket', 'CNN_model.h5')
    model = load_model('Streamlit/local_CNN_model.h5')

    # Upload image
    file_uploaded = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)

    # Classify button
    if st.button("Classify"):
        if file_uploaded is None:
            st.write("Please upload an image to classify.")
        else:
            with st.spinner('Classifying...'):
                # model = load_model_from_s3('larissahuang-bstn-bucket', 'CNN_model.h5')
               
                if model is not None:
                    predictions = predict(image, model)
                    st.success('Classification complete!')
                    st.write(predictions)

if __name__ == "__main__":
    main()