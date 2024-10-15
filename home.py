import streamlit as st
import boto3
import json
from PIL import Image
import io
import random
import base64
import time

# Page Config
st.set_page_config(
    page_title="Image Geneneration & Evaluation",
    layout="wide"
)

# CSS Fixes
st.markdown("""
    <style>
        header { display: none !important; }
        .block-container { padding-top: 0; }
        .st-emotion-cache-1v0mbdj { margin: 0 auto; }
    </style>
""", unsafe_allow_html=True)


st.sidebar.image("images/logo.png", width=80)
st.sidebar.header("Image Generation & Evaluation")

st.sidebar.write(
    """
    Use Amazon Bedrock to generate images 
    and automatically evaluate them.
    """
)

url = "mailto:tavaraul@amazon.co.uk"

st.sidebar.write(
"Feedback/Suggestions? [Get in Touch!](%s)" % url
)

# List of Stable Diffusion Preset Styles
sd_presets = [
    "3d-model",
    "analog-film",
    "cinematic",
    "digital-art",
    "enhance",
    "neon-punk",
    "photographic",
]

# List of Image Generator model names
model_names = ["Stable Diffusion SDXL 1.0", "Titan Image Generator G1 v2"]

# Setup bedrock
bedrock_client = boto3.client( service_name="bedrock-runtime", region_name="us-east-1" )

# Bedrock call to Stable Diffusion
def generate_image_sd(text):

    style = random_choice()

    print(f"Using Stable Diffusion, style as {style}")

    body = {
        "text_prompts": [{"text": f"Create a featured image for the following prompt. Be as real as possible with your generation. Prompt: {text}"}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 45,
        "style_preset": style
    }

    body = json.dumps(body)

    modelId = "stability.stable-diffusion-xl-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    results = response_body.get("artifacts")[0].get("base64")
    return results

# Bedrock call to Amazon Titan Image Generator G1 v2
def generate_image_at(prompt):

    print("Using Amazon Titan Image Generator G1 v2")

    body = json.dumps({
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,
                "negativeText": "bad quality, low res"
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,
                "height": 1024,
                "width": 1024,
                "cfgScale": generate_random_double(),
                "seed": generate_random_int()
            }
    })

    model_id = 'amazon.titan-image-generator-v2:0'
    accept = "application/json"
    content_type = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())

    finish_reason = response_body.get("error")

    if finish_reason is not None:
        print(f"Image generation error. Error is {finish_reason}")

    generated_image = response_body.get("images")[0]

    return generated_image

# Bedrock API call to Claude 3.5 Sonnect for image relevance evaluation
def evaluate_image(prompt, image_base64):

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            },
                        },
                        {"type": "text", 
                        "text": ( f"Evaluate the attached image against the original prompt: {prompt}. "
                                                    "Provide a short description of the generated image, give it score from 1 to 10, "
                                                    "share the reason of your scoring (maximum 15 words), and suggest how the picture could better attain to the original prompt (maximum 20 words). "
                                                    "Format your response like the following JSON structure: {{\"description\": \"x\", \"score\": \"y\", \"reason\": \"z\", \"suggestions\": \"w\"}} " )
                        },
                    ],
                }
            ],
        }
    )

    # modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    modelId = "anthropic.claude-3-sonnet-20240229-v1:0"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    
    response_body = json.loads(response.get("body").read())
    
    evaluation_text = response_body["content"][0]["text"]
    
    return evaluation_text

# Aux Methods
def random_choice():

    rdn_option = random.choice(sd_presets)

    # remove that random choice from the list
    sd_presets.remove(rdn_option)

    return rdn_option

# Random Double
def generate_random_double(min_value=5.0, max_value=9.0):
    return random.uniform(min_value, max_value)

# Random Int
def generate_random_int(min_value=1, max_value=50):
    
    return random.randint(min_value, max_value)

# Turn base64 string to image with PIL
def base64_to_pil(base64_string):
    
    import base64

    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    return image


# Frontend Piece
st.markdown("# Image Generation & Evaluation Demo")

st.subheader("How it works?")

st.write(
    """
    This demo showcases the capabilities of Amazon Bedrock's Stable Diffusion SD XL 1.0 and Amazon Titan Image Generator G1 v2 models.
    Generate images from text prompts and automatically evaluate their relevance based on the provided prompt using Anthropic's Claude 3 Sonnet model .
    """
)

st.subheader("Demo Playground")

st.write(
    """
    Use the form below to generate and evaluate images based on your prompt.
    """
)

# Create form
form = st.form("image_generation_form", clear_on_submit=True)

with form:

    # Value for the number of images to generate
    range_value = st.selectbox(
        "How many images you would like to generate?",
        (1,2,3)
    )

    # Model selection
    selected_model = st.selectbox("Select the Image Generation model:", model_names)

    # Text input for prompt
    prompt = st.text_area("Enter Prompt:")

    # Add the submit button
    submitted = form.form_submit_button("Generate & Evaluate")


# Initialize session state for images and evaluations
if "images" not in st.session_state:
    st.session_state.images = []
if "evaluations" not in st.session_state:
    st.session_state.evaluations = []


# Generate images from prompt
if submitted:

    if prompt:

        with st.spinner(""):

            st.session_state.images = []
            st.session_state.evaluations = []

            status_text = st.empty()    

            for i in range(range_value):  # Generate x images

                status_text.text(f"Generating image {i+1} of {range_value} ...")

                image_base64 = generate_image_sd(prompt) if selected_model == model_names[0] else generate_image_at(prompt)

                if image_base64:
                    image_pil = base64_to_pil(image_base64)
                    st.session_state.images.append((image_pil, image_base64))
            
            # After the loop, update the placeholder with a final message
            status_text.text("Generation complete!")

        # Evaluate images for relevance
        if st.session_state.images:
            with st.spinner(""):

                eval_text = st.empty()

                for idx, (image_pil, image_base64) in enumerate(st.session_state.images):
                    
                    eval_text.text(f"Evaluating image {idx + 1} of {len(st.session_state.images)}")

                    evaluation = evaluate_image(prompt, image_base64)
                    st.session_state.evaluations.append(evaluation)
                
                eval_text.text("Evaluation complete!")

# Display images and evaluations in columns
if st.session_state.images:

    columns = st.columns(range_value) 

    for idx, (image_pil, _) in enumerate(st.session_state.images):

        col = columns[idx % range_value] 

        with col:

            st.image(image_pil)

            if st.session_state.evaluations:
                evaluation = st.session_state.evaluations[idx]
                evaluation_json = json.loads(evaluation)
                st.write(f"Description: {evaluation_json['description']}")
                st.write(f"Score: {evaluation_json['score']}")
                st.write(f"Reason: {evaluation_json['reason']}")
                st.write(f"Suggestions: {evaluation_json['suggestions']}")
