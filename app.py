import streamlit as st
from backend import transform_glue_etl_to_databricks_etl

# Page Configuration
st.set_page_config(page_title='Code Converter', layout='wide', page_icon=":gear:")

# Custom Styling for a Sleek UI
light_theme_css = """
    <style>
        body, .stApp {
            background-color: #d0eaff;
            color: #ffffff;
        }
        .stTextInput, .stSelectbox, .stFileUploader, .stTextArea {
            background-color: #b0d4ff;
            color: #ffffff;
            border-radius: 5px;
        }
        .stButton>button {
            background-color: #0088cc;
            color: white;
            border-radius: 5px;
            font-weight: bold;
        }
        .centered-title {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            color: #004a80;
        }
        .login-title {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: #004a80;
        }
        .stMarkdown h2 {
            text-align: center;
            color: #004a80;
        }
    </style>
"""
st.markdown(light_theme_css, unsafe_allow_html=True)

# Layout - Logo and Title
col1, col2, col3 = st.columns([1, 5, 1])
with col1:
    try:
        st.image("./genpact-logo.png", use_container_width=True)
    except FileNotFoundError:
        st.warning("⚠️ Logo not found. Please ensure the file is in the correct location.")

with col2:
    st.markdown("<h2 class='centered-title'>Code Converter</h2>", unsafe_allow_html=True)

# Authentication
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<h2 class='login-title'>🔐 Login</h2>", unsafe_allow_html=True)
    with st.form("LoginForm", clear_on_submit=False):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("Login")
        if submitted:
            if username == "admin" and password == "password":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")

def logout():
    st.session_state.logged_in = False
    st.rerun()

if not st.session_state.logged_in:
    login()
    st.stop()

with col3:
    st.button("🚪 Logout", on_click=logout)

# Main Interface
home, help = st.tabs(["🏠 Home", "❓ Help"])

with home:
    st.subheader("🔄 Code Conversion")
    col1, col2 = st.columns(2)
    with col1:
        from_option = st.selectbox("Convert From:", ['AWS EMR PySpark','AWS Glue PySpark','AWS Sagemaker ML PySpark',
                                'Text','SQL','Databricks PySpark','Databricks ML PySpark'], index=0)
    with col2:
        to_option = st.selectbox("Convert To:", ['AWS EMR PySpark','AWS Glue PySpark','AWS Sagemaker ML PySpark',
                                'Text Explaination','SQL','Databricks PySpark','Databricks ML PySpark'], index=1)

    uploaded_file = st.file_uploader("📂 Upload a Python (.py) File", type="py")
    code_input = st.text_area("Paste Your Code Here", height=200, placeholder="Paste your Python or SQL code...")

    if st.button("🚀 Transform Code", use_container_width=True):
        with st.spinner("🔄 Processing Code..."):
            try:
                input_code = uploaded_file.read().decode("utf-8") if uploaded_file else code_input
                if input_code:
                    output_response = transform_glue_etl_to_databricks_etl(from_option, to_option, input_code)
                    st.subheader("✅ Transformed Code:")
                    st.code(output_response, language="python")
                else:
                    st.warning("⚠️ Please upload a file or paste code before converting.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

with help:
    st.subheader("📖 Help & Documentation")