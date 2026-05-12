import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import cv2

classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
st.sidebar.markdown("""
## About

This application is deployed using Streamlit and a GitHub repository.

The uploaded image is processed before being fed to the model so that it resembles the EMNIST dataset images on which the model was trained.

You can also view the processed image in case recognition does not work properly, allowing you to adjust thickness, background, orientation, or lighting.

**Author:** Rageshwer Singh\n 
MSc(Data Science and AI)\n
LinkedIN www.linkedin.com/in/rageshwer-singh-06a178384\n
GitHub https://github.com/rageshwer?tab=repositories
""")
st.title("Handwritten Digit/Alphabet Recognition Application")
name=st.text_input("Enter your name:")
if name=='Vidhi Kala' or name=='vidki kala' or name=='vidhi' or name=='Vidhi' or name=='VIDHI' or name=='VIDHI KALA':
    st.write("Oh My God!! Are you seriously THE VIDHI KALA, the one my maker is always talking about?")
    check=st.radio('Seriously are you that Vidhi?',['No, I am not that lucky','Yes, I am'])
    if check=='Yes, I am':
        st.success("Finally I meet the one.")
        st.write("He talks about you all the time, sometimes I am like, Wow! how can someone be that pretty. Are you the prettiest lady in the world?")
        button1=st.button('Yeah I guess so')
        button2=st.button("Oh, definitely I am")
        if button1: st.write("Hmmmm!!")
        if button2:
            st.write("Woah! You sure are confident. But then, If someone is as pretty as you I guess they are allowed to be. Anyways, ")
            st.subheader(f"Hey {name}! This is a streamlit application that uses a CNN to recognize the handwritten digit or alphabet.")

else :
    st.subheader(f"Hey {name}! This is a streamlit application that uses a CNN to recognize the handwritten digit or alphabet.")

st.subheader("Instructions for uploading image:")
st.text(f"The character written should have appropriate width (uniform). So {name}, try to minimise the shadows and extra objects in the image. The best results are obtained when written on a blank white paper with a thickness of a marker pen.")
x_test=None
test_image1=st.camera_input("Click a picture of handwritten character :")
st.subheader("OR")
test_image=st.file_uploader("Upload image of the handwritten character:",type=['jpg','jpeg','png'])


def preparation(image_):
    image1=image.load_img(image_,color_mode='grayscale',target_size=(28,28))
    image_arr=np.array(image1)
    # invert colors
    img_inv=255-image_arr
    # convert to black and white
    _,img2=cv2.threshold(
        img_inv,100,255,cv2.THRESH_BINARY
    )
    # now we can create a bounding rectangle on the non zeros, but it captures all the noise pixels also
    # so we find the largest connected component and only keep the bounding rectangle around it.

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img2,connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean = np.zeros_like(img2)
    clean[labels == largest_label] = 255

    coords = cv2.findNonZero(clean)
    x, y, w, h = cv2.boundingRect(coords)

    # now the part of centering and cropping

    # Crop character
    cropped = clean[y:y+h, x:x+w]

    # Resize while maintaining aspect ratio
    if w > h:
        new_w = 20
        new_h = int(h * (20 / w))
    else:
        new_h = 20
        new_w = int(w * (20 / h))

    resized = cv2.resize(cropped, (new_w, new_h))

    #Create final 28x28 black image
    final = np.zeros((28, 28), dtype=np.uint8)

    # Compute center offsets
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    # Put resized digit in center
    final[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    # normalization 
    img_norm=final/255.0
    img = img_norm
    canvas = img.reshape(1,28,28,1)

    

    return canvas



if test_image is not None:
    x_test=preparation(test_image)
    from prediction import pred
    index=pred(x_test)
    col1,col2=st.columns(2)
    with col1:
        st.image(test_image,width=200)
        st.write("Image Uploaded")
    with col2:
        st.image(x_test,width=200)
        st.write("Image fed to model after processing")
    st.success(f"The entered handwritten character is {classes[index]}")

elif test_image1 is not None:
    x_test=preparation(test_image1)
    from prediction import pred
    index=pred(x_test)
    col1,col2=st.columns(2)
    with col1:
        st.image(test_image1,width=200)
        st.write("Image Uploaded")
    with col2:
        st.image(x_test,width=200)
        st.write("Image fed to model after processing")
    st.success(f"The entered character is {classes[index]}")


else : st.write("Please upload from any source above. Keep the character at the center. You can look at the image uploaded and the processed image that is fed to the CNN model.")

# during training, the images are rotated, flipped and white on black background. So our model learns accordingly.
# hence we do the same to the input image.