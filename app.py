import streamlit as st
import requests
from PIL import Image
import base64
import io

# Title of the app
st.title("Product Information Extractor")

# Access API keys from st.secrets
openai_api_key = st.secrets.get("openai_api_key", "")
perplexity_api_key = st.secrets.get("perplexity_api_key", "")

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# Function to extract product name using OpenAI Vision API
def extract_product_name_from_image(image, openai_api_key):
    img_str = encode_image(image)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",  # Updated model name as per your instruction
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the product name from this image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_str}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        product_name = result['choices'][0]['message']['content'].strip()
        return product_name
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

# Function to get product information using Perplexity API
def get_product_info(product_name, perplexity_api_key):
    url = "https://api.perplexity.ai/chat/completions"

    headers = {
        "Authorization": f"Bearer {perplexity_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "llama-3.1-sonar-huge-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "As an expert skincare assistant, provide the detailed ingredients for the given product."
            },
            {
                "role": "user",
                "content": f"Please provide the detailed ingredients of {product_name}. No need markdown format."
            }
        ],
        "max_tokens": 1500,  # Reduced from 1000 to limit output length
        "temperature": 0.7,  # Increased for more creative responses
        "top_p": 0.9,
        "return_citations": True,
        "search_domain_filter": [],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "year",
        "top_k": 10,
        "stream": False,
        "frequency_penalty": 0.8   # Increased to penalize token repetition
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            result = response.json()
            # Extract the content from the assistant's message
            perplexity_text = result['choices'][0]['message']['content']
            return perplexity_text
        except (ValueError, KeyError, IndexError) as e:
            st.error("Failed to parse response.")
            st.write(f"Exception: {e}")
            st.write("Response content:", response.text)
            return None
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

# Check if API keys are provided
if not openai_api_key:
    st.error("OpenAI API key not found in st.secrets.")
elif not perplexity_api_key:
    st.error("Perplexity API key not found in st.secrets.")
else:
    # Upload image file
    uploaded_file = st.file_uploader("Upload an image of the product", type=["jpg", "jpeg", "png"])

    # Input product name directly
    product_name_input = st.text_input("Or, enter the product name directly")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Extract product name from image using OpenAI Vision API
        with st.spinner('Extracting product name using OpenAI Vision API...'):
            product_name = extract_product_name_from_image(image, openai_api_key)

        if product_name and "no product name" not in product_name.lower():
            st.success(f"Product Name: {product_name}")
            # Get product information using Perplexity API
            with st.spinner('Retrieving product information from Perplexity AI...'):
                product_info = get_product_info(product_name, perplexity_api_key)

            if product_info:
                st.text_area("Product Ingredients and Information", value=product_info, height=300)
            else:
                st.error("Error retrieving data from Perplexity API.")
        else:
            st.warning("Could not extract product name. Please upload another image showing the product name.")

    elif product_name_input.strip() != "":
        product_name = product_name_input.strip()
        st.success(f"Product Name: {product_name}")
        # Get product information using Perplexity API
        with st.spinner('Retrieving product information from Perplexity AI...'):
            product_info = get_product_info(product_name, perplexity_api_key)

        if product_info:
            st.text_area("Product Ingredients and Information", value=product_info, height=300)
        else:
            st.error("Error retrieving data from Perplexity API.")

    else:
        st.info("Please upload an image or enter the product name.")
