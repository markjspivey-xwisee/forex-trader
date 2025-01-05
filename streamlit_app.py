import streamlit as st
import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

st.set_page_config(page_title='AI-Powered Forex Trading Agent', layout='wide')

def main():
    # Title and logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://raw.githubusercontent.com/microsoft/fluentui-emoji/main/assets/Robot/3D/robot_3d.png", width=100)
    with col2:
        st.title('AI-Powered Forex Trading Agent')
    
    st.write("Setting up deployment structure...")
    
    # Display environment info
    st.subheader("Environment Information")
    st.code(f"""
    Python Path: {sys.path}
    Current Directory: {current_dir}
    Working Directory: {os.getcwd()}
    """)

if __name__ == '__main__':
    main()
