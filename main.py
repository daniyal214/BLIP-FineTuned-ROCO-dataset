import streamlit as st
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

# Load the fine-tuned model and tokenizer
output_dir = "saved_model/chest_xrays_1"
tokenizer = BlipProcessor.from_pretrained(output_dir)
model = BlipForConditionalGeneration.from_pretrained(output_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


# Function to generate caption from image
def generate_caption(image):
    # Preprocess the image
    image = Image.open(image).convert("RGB")
    image = image.resize((100, 100), Image.ANTIALIAS)

    # Convert the image to input tensor
    inputs = tokenizer(images=image, return_tensors="pt", padding=True)

    # Move inputs to the appropriate device
    for key in inputs:
        inputs[key] = inputs[key].to(device)

    # Generate the caption
    generated_ids = model.generate(**inputs)
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # print("Image Path:", image_path)
    print("Generated Caption:", generated_caption)

    return generated_caption


# @st.cache(allow_output_mutation=True)
# def initialize_object(image):
#     # Put your object initialization code here
#     obj = generate_caption(image)
#     return obj


# Streamlit app
def main():
    st.title("Image Captioning with Fine-Tuned Model")
    st.write("Upload an image to get its caption!")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Display the uploaded image
        # image = Image.open(uploaded_file)
        # image = image.resize((100, 100), Image.ANTIALIAS)
        # st.image(image, caption="Uploaded Image", use_column_width=True)

        # Generate caption
        caption = generate_caption(uploaded_file)
        st.write("Generated Caption:")
        st.success(caption)


if __name__ == "__main__":
    main()
