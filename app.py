import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from keras.models import load_model
import pandas as pd
from streamlit_image_select import image_select

def predict(_image, _model):
    # Convert image to RGB
    if _image.mode != 'RGB':
        _image = _image.convert('RGB')

    test_image = tf.keras.preprocessing.image.img_to_array(_image.resize((224, 224)))
    test_image = preprocess_input(test_image)  # Ensure this matches model's preprocessing
    test_image = np.expand_dims(test_image, axis=0)
    
    scores = _model.predict(test_image)
    class_names = ['AFRICAN PIED HORNBILL',
    'AMERICAN DIPPER',
    'AMERICAN WIGEON',
    'ASHY STORM PETREL',
    'ASIAN GREEN BEE-EATER',
    'ASIAN OPENBILL STORK',
    'AUCKLAND SHAQ',
    'AUSTRALASIAN FIGBIRD',
    'BANDED BROADBILL',
    'BLACK VENTED SHEARWATER',
    'BLUE GRAY GNATCATCHER',
    'BLUE MALKOHA',
    'BLUE THROATED PIPING GUAN',
    'BROWN HEADED COWBIRD',
    'CAMPO FLICKER',
    'CASPIAN TERN',
    'COPPERSMITH BARBET',
    'CRESTED WOOD PARTRIDGE',
    'CRIMSON SUNBIRD',
    'D-ARNAUDS BARBET',
    'DARK EYED JUNCO',
    'DUNLIN',
    'EASTERN MEADOWLARK',
    'EASTERN YELLOW ROBIN',
    'FRILL BACK PIGEON',
    'GREAT ARGUS',
    'GREATER PRAIRIE CHICKEN',
    'GREY HEADED CHACHALACA',
    'HOUSE FINCH',
    'INDIAN PITTA',
    'JACOBIN PIGEON',
    'KNOB BILLED DUCK',
    'LAUGHING GULL',
    'LIMPKIN',
    'LOGGERHEAD SHRIKE',
    'MARABOU STORK',
    'MCKAYS BUNTING',
    'MERLIN',
    'MILITARY MACAW',
    'NORTHERN PARULA',
    'ORANGE BREASTED TROGON',
    'ORNATE HAWK EAGLE',
    'OVENBIRD',
    'OYSTER CATCHER',
    'PALM NUT VULTURE',
    'PHAINOPEPLA',
    'PLUSH CRESTED JAY',
    'PYRRHULOXIA',
    'RAZORBILL',
    'RED BEARDED BEE-EATER',
    'RED BILLED TROPICBIRD',
    'RED CROSSBILL',
    'RED KNOT',
    'RED TAILED HAWK',
    'ROSE BREASTED COCKATOO',
    'ROSEATE SPOONBILL',
    'RUFOUS TREPE',
    'RUFUOS MOTMOT',
    'SAYS PHOEBE',
    'SNOW GOOSE',
    'SNOWY SHEATHBILL',
    'SORA',
    'STRIATED CARACARA',
    'SWINHOES PHEASANT',
    'TAWNY FROGMOUTH',
    'VARIED THRUSH',
    'VEERY',
    'VIOLET BACKED STARLING',
    'VIOLET GREEN SWALLOW',
    'WILLOW PTARMIGAN',
    'WOOD DUCK',
    'WOOD THRUSH',
    'WOODLAND KINGFISHER',
    'WRENTIT',
    'YELLOW BREASTED CHAT']
    
    result = f"The selected image belongs to the species {class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(4) } % confidence." 
    return result

def main():
    st.title('Bird Species Classifier')
    st.markdown("Larissa Huang - BrainStation Data Science Capstone Project.")
    st.subheader("Either select an image or scroll down to upload your own image, then click the 'Classify' button.")
    st.markdown("The model will predict the bird's species and return its degree of confidence.")
    
    #load model
    _model = load_model('Streamlit/hi-acc.h5')
    # Load the CSV file
    file_paths = "Sprint_4/df_test_75_streamlit.csv"
    df = pd.read_csv(file_paths)

    # Extract values and make arrays for img-select
    path_array = df['path'].to_numpy()
    species = df['species'].values

    # stringify species values for captions array
    caption_array = [str(value) for value in species] 
 
    #streamlit-image-select ui image picker
    img = image_select(
        label="Select a bird",
        images=path_array,
        captions=caption_array,
    )   
    
   # loop through df with image paths and species names
    for index, row in df.iterrows():
        image_path = row['path'] 
        species = row['species']
        
        # Check if the current image path matches the selected image path
        if image_path == img:
            # Get the corresponding caption for the selected image
            selected_caption = species
            break  # Exit the loop once the selected image is found
        
    st.subheader("Your selection:")
    if img is not None:
        #open the image
        _image = Image.open(img)
        #show the image with selected caption
        st.image(_image, caption=selected_caption, use_column_width=True)
       
    st.subheader("Upload your image:") 
    #upload own image
    file_uploaded = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
    if file_uploaded is not None:
        _image = Image.open(file_uploaded)
        st.image(_image, caption='Uploaded Image', use_column_width=True)

    # Classify button
    if st.button("Classify"):
        if img is None:
            st.write("Please upload an image to classify.")
        else:
            with st.spinner('Classifying...'):
                if _model is not None:
                    predictions = predict(_image, _model)
                    st.success('Classification complete!')
                    st.write(predictions)

if __name__ == "__main__":
    main()