import streamlit as st
import requests
from PIL import Image
import base64
import io

# Title of the app
st.title("Product Information Extractor")

# Sidebar for API key inputs
st.sidebar.header("API Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
perplexity_api_key = st.sidebar.text_input("Perplexity API Key", type="password")

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
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {
                "role": "system",
                "content": "Be precise and concise."
            },
            {
                "role": "user",
                "content": f"Output the detailed ingredients and the information of the ingredients of {product_name}."
            }
        ],
        "max_tokens": 500,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": False,
        "search_domain_filter": [],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        perplexity_text = result.get("completion", "No information found.")
        return perplexity_text
    else:
        st.error(f"Error {response.status_code}: {response.text}")
        return None

# Upload image file
uploaded_file = st.file_uploader("Upload an image of the product", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Check if API keys are provided
    if not openai_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not perplexity_api_key:
        st.error("Please enter your Perplexity API key in the sidebar.")
    else:
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
