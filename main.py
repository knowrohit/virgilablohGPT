


import streamlit as st
from streamlit_chat import message as st_message
from gtts import gTTS
from IPython.display import Audio
import transformers
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig

@st.cache_resource
def get_models():
    # tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/bernard/GLM_tokenizer", trust_remote_code=True)
    # model = AutoModel.from_pretrained("/content/drive/MyDrive/bernard/GLM_model", trust_remote_code=True).half().cuda()
    model = AutoModelForCausalLM.from_pretrained("/content/drive/MyDrive/bernard/mpt7b_model", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("/content/drive/MyDrive/bernard/mpt7b_tok")
    name = 'mosaicml/mpt-7b-chat'

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    config.attn_config['attn_impl'] = 'triton'
    config.init_device = 'cuda:0' # For fast initialization directly on GPU!
    config.max_seq_len = 4096 # (input + output) tokens can now be up to 4096


    model = transformers.AutoModelForCausalLM.from_pretrained(
      name,
      config=config,
      torch_dtype=torch.bfloat16, # Load model weights in bfloat16
      trust_remote_code=True
    )
    model.eval()
    return tokenizer, model

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Set Streamlit configuration
st.set_page_config(page_title="Fashion Concierge", layout="wide")

# Topics related to fashion
FASHION_TOPICS = [ 
    "Open chat",
    "", # string for generic discussion
    "summer fashion", 
    "winter trends", 
    "vintage style", 
    "street style", 
    "workout gear", 
    "accessories", 
    "sustainable fashion", 
    "latest fashion tech",
    "fashion influencers",
    "personal styling tips",
    "dress codes"
]

TASKS = {
    "Generate code :": "write code",
    "Summarize :": "summarize the content",
    "Sentiment analysis :": "sentiment analysis",
    "Act as a Friend :": "be a friend",
    "Act as a Teacher :": "be a teacher",
    "Tell me a joke !": "do comedy and crack jokes",
    "Generate short stories": "tell short stories",
    "Translate the text : ": "translate text",
    "Write in detail ": "create a piece of writing",
    "Verify the information :": "verify information",
    "Give me advice: ": "give advice",
    "Forecast the prompt/content ": "predict future trends",
    "Analyze the data : ": "analyze data",
    "Generate/Brainstorm ideas : ": "generate ideas",
    "Negotiate a deal : ": "negotiate a deal",
    "Provide motivation : ": "provide motivation",
}
TASK_MAX_LENGTH = {
    "Generate code :": 350,
    "Summarize :": 100,
    "Sentiment analysis :": 200,
    "Act as a Friend :": 75,
    "Education": 100,
    "Humor": 70,
    "Storytelling": 350,
    "Translation": 70,
    "Writing": 300,
    "Fact-checking": 250,
    "Advice": 200,
    "Forecast": 200,
    "Analysis": 200,
    "Brainstorming": 250,
    "Negotiation": 100,
    "Motivation": 50,
}


# Start of the Streamlit app
st.title("🥷🏻 Virgil Abloh: AI-Powered Fashion Recommender System")
st.header("I'm always trying to prove to my 17-year-old self that I can do creative things I thought weren't possible.")

# Sidebar for chat configuration
st.sidebar.title("Chat Configuration")
topic = st.sidebar.selectbox("Choose a topic", FASHION_TOPICS)
task = st.sidebar.selectbox("Choose a task", list(TASKS.keys()))




def predict(user_input, topic, task):
    tokenizer, model = get_models()
    input_text = f"{topic} {TASKS[task]} {user_input}"
    
    # Tokenize the input text
    input_tokens = tokenizer.encode(input_text, return_tensors="pt")

    # Move input_tokens to GPU
    input_tokens = input_tokens.to('cuda')

    max_length = TASK_MAX_LENGTH.get(task, 100)  # default to 200 if task is not found in the dictionary

    # Generate the model's output
    output = model.generate(input_tokens, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)


    # Ignore the input tokens in the output
    output = output[:, input_tokens.shape[-1]:]

    # Decode the generated tokens
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response

def generate_answer():
    user_input = st.session_state.user_input
    # Generate a response using the predict function
    response = predict(user_input, topic, task)
    # Convert text to speech
    if task == "Coding":
        response = f"```\n{response}\n```"
    else:
        # Convert text to speech
        tts = gTTS(text=response, lang='en')
        tts.save("/content/response.mp3")
    
        audio_file = open('/content/response.mp3', 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/mp3')


    st.session_state.chat_history.append({"message": user_input, "is_user": True})
    st.session_state.chat_history.append({"message": response, "is_user": False})


st.text_input("You: ", key="user_input", on_change=generate_answer)

for i, chat in enumerate(st.session_state.chat_history):
    st_message(**chat, key=str(i)) #unpacking

# Clear chat history
if st.button("Clear"):
    st.session_state.chat_history = []


