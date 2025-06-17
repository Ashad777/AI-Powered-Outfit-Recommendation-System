import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Cohere
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("COHERE_API_KEY")

# Initialize LLM
cohere_llm = Cohere(cohere_api_key=api_key)

outfit_prompt = PromptTemplate(
    input_variables=["weather", "occasion", "colors", "gender", "style"],
    template="""
You are an AI fashion stylist.

The user gave:
- Weather: {weather}
- Occasion: {occasion}
- Favorite Colors: {colors}
- Gender: {gender}
- Style Preference: {style}

If anything is unclear, ask clarifying questions.
If everything is clear, suggest a complete outfit including:
- Top
- Bottom
- Outerwear
- Accessories
- Shoes

Keep the style fashionable, practical, and relevant to the context.
"""
)

outfit_chain = LLMChain(llm=cohere_llm, prompt=outfit_prompt)


st.set_page_config(
    page_title="ToadStyle",
    page_icon="logo.png",  
    layout="centered"
)


st.title("ToadStyle – AI-Powered Outfit Recommendation System 🤖")

st.markdown("Let me help you pick the perfect outfit for any weather or occasion!")


gender = st.selectbox("👤 Select Gender", ["Male", "Female", "Unisex"])
style = st.selectbox("🧵 Preferred Style", ["Classic", "Modern", "Streetwear", "Minimal", "Formal"])
weather = st.text_input("🌦️ Enter the weather (e.g., Cold, Hot, Rainy)")
occasion = st.text_input("🎉 What's the occasion? (e.g., Wedding, Party, Work)")
colors = st.text_input("🎨 Favorite colors (comma-separated, e.g., Black, White, Blue)")


if st.button("Try Example: Winter Wedding Look"):
    gender = "Male"
    style = "Classic"
    weather = "Cold"
    occasion = "Wedding"
    colors = "White"


if st.button("✨ Get Outfit Recommendation"):
    if not weather or not occasion or not colors:
        st.warning("Please fill in all the fields.")
    else:
        with st.spinner("Generating your outfit..."):
            try:
                outfit = outfit_chain.run(
                    weather=weather,
                    occasion=occasion,
                    colors=colors,
                    gender=gender,
                    style=style
                )
                st.success("Here's your personalized outfit:")
                st.markdown(f"🪄 **Recommendation:**\n\n{outfit}")

                st.download_button(
                    label="📥 Download Recommendation",
                    data=outfit,
                    file_name="outfit_recommendation.txt",
                    mime="text/plain"
                )

            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer with custom styling
st.markdown("""
    <hr style="margin-top: 50px; border-top: 1px solid #bbb;" />
    <div style="text-align: center; color: gray; font-size: 14px; padding-top: 10px;">
        🚀 Built by <b>Ashad K</b> · Powered by <b>LangChain</b>, <b>Cohere</b>, and <b>Streamlit</b><br>
        <a href="https://github.com/Ashad777/AI-Powered-Outfit-Recommendation-System" target="_blank" style="color: #888;">View on GitHub</a>
    </div>
""", unsafe_allow_html=True)
