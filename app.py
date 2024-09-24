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
                "content": "As an expert assistant, your task is to find and provide the detailed ingredients and information of the ingredients for the given product. If you cannot find the exact ingredients, provide as much relevant information as possible."
            },
            {
                "role": "user",
                "content": f"Please provide the detailed ingredients and information of the ingredients of {product_name}."
            }
        ],
        "max_tokens": 1500,
        "temperature": 0.2,
        "top_p": 0.9,
        "return_citations": True,
        "search_domain_filter": [],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "all",  # Changed from "month" to "all"
        "top_k": 10,  # Increased from 0 to 10
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0  # Changed from 1 to 0
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            result = response.json()
            st.write(result)  # For debugging purposes
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


# Upload image file
uploaded_file = st.file_uploader("Upload an image of the product", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Check if API keys are provided
    if not openai_api_key:
        st.error("OpenAI API key not found in st.secrets.")
    elif not perplexity_api_key:
        st.error("Perplexity API key not found in st.secrets.")
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

