import yolov5
import os
from PIL import Image
import cv2
import streamlit as st
import time
from datetime import datetime
from urllib.parse import urlparse
from posixpath import basename, dirname
import img2pdf

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def predict(image_path=None,model_path=r"C:\Users\indra\Downloads\best.pt"):

    model = yolov5.load(model_path)
    

    # set model parameters
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = True  # NMS multiple labels per box
    model.max_det = 1000  # maximum number of detections per image


    
    results = model(image_path)

    home_dir = os.getcwd()
    
    
    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    p = home_dir+"/new_pred" +str(current_time).replace(":","_")
    os.mkdir(p)
    results.save(save_dir=p)
    print("Saved to: {}".format(p))

    
    url = image_path
    parse_object = urlparse(url)
    #im_path = parse_object.base
    saved_image_path = p+"2" + "/" + basename(parse_object.path)
    print("Reading image from dir: {}".format(saved_image_path))
    
    files = Image.open(saved_image_path)
    return files
    
    
    """
    #results.save(save_dir=save_path)


    #open_path = os.path.join(save_path,".jpg")
    #path = open_path.rstrip("\\.jpg")
    #new_path = path + ".jpg"
    #files = Image.open(new_path)

    #return files
    """
    """
    b = str(image_path).split("/")[-1]
    path = os.path.join(save_path , b)
    #n = path.replace("\\","'\'")
    i = cv2.imread(path)
    cv2.imshow(i)
    """


#f = predict(image_path="https://img.freepik.com/free-photo/woman-with-medical-mask-out-sale-shopping-spree-with-shopping-bag_23-2148673251.jpg")

#f.show()

st.title("Mask Detector Application  :mask:")
st.image(image="https://wallpaperaccess.com/full/8417231.jpg")
st.write("This app uses yolov5 model for object detection. Only images are supported for now, videos will be supported soon!")

path = st.text_input(label="Please enter the image url: ")

if st.button(label="Start Service!"):
    with st.progress(value=0,text="Starting the Vision Service..."):
        time.sleep(2)
        st.progress(value=10,text="Fetching the image from url...")
        
        f = predict(image_path=path)
        st.progress(value=25,text="Vision engine running...")
        time.sleep(2)
        st.progress(value=50,text="Generating bounding boxes...")
        time.sleep(1)
        st.progress(value=75,text="Inference complete! ...")
        st.progress(value=90,text="Generating image...")
        # converting to jpg
        rgb_im = f.convert("RGB")
        rgb_im.save("test_new.jpg")
        img = Image.open("test_new.jpg")



        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        #pdf_bytes = img2pdf.convert("test.jpg")

        with open("name.pdf","wb") as files_new:
            files_new.write(img2pdf.convert("test.jpg"))



        st.progress(value=100,text="Completed! :100:")
        
        time.sleep(2)
        st.success("Objects have been detected successfully! :beers:")
        time.sleep(2)
        st.download_button(label="Click here to download", data=byte_im, file_name="pred.png",mime="image/jpeg")
        time.sleep(3)
        
        time.sleep(5)
        st.success("Service generated inference successfully! :beers:")
        time.sleep(10)
        