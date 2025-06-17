# 🤖 AI-Powered Outfit Recommendation System

This project is an intelligent web-based application that provides personalized outfit recommendations based on user preferences. It uses a Large Language Model (LLM) from Cohere to generate fashion suggestions tailored to the user's gender, weather conditions, favorite colors, and occasion.

The system was developed using **Streamlit** for the frontend interface and integrates **LangChain** and **Cohere API** on the backend to handle prompt templating and natural language generation.

🔗 **Live Demo:** [ToadStyle – AI-Powered Outfit Stylist](https://toadstyle.streamlit.app/)

---

## 🧠 What I Did

- Developed a user-friendly web application using **Streamlit**.
- Integrated **Cohere's LLM** via **LangChain** to dynamically generate outfit suggestions in natural language.
- Created a structured prompt template that passes user inputs (gender, weather, occasion, colors, style) to the LLM.
- Handled API responses and formatted them into clean, readable fashion advice.
- Added custom UI elements including logo, page icon, and themed colors for a polished look.
- Deployed the app and prepared it for demonstration or production use.

---

## 💡 Key Features

- **Gender-Specific Recommendations**: Personalized outfits for male, female, or unisex options.
- **Weather Adaptation**: Smart clothing suggestions for hot, cold, rainy, or any other climate.
- **Occasion Awareness**: Outfits tailored for events like weddings, parties, or casual wear.
- **Color Customization**: Takes your favorite colors into account for aesthetic alignment.
- **Natural Language Output**: Clean, human-like fashion suggestions written by an LLM.
- **Clean and Intuitive UI**: Built entirely with Streamlit and customized for a smooth experience.

---

## 🛠️ Technologies Used

- **Python** – Core language for logic and integration
- **Streamlit** – Frontend web framework
- **Cohere API** – LLM for generating outfit suggestions
- **LangChain** – Manages prompts and LLM communication
- **.env Config** – Securely stores API keys
- **Custom Theme** – Includes logo, favicon, and color styling

---

## 📌 How It Works

1. The user enters five inputs:
   - Gender
   - Current weather
   - Occasion
   - Favorite colors
   - Preferred style

2. These values are passed into a structured prompt template.

3. The prompt is sent to the **Cohere LLM** using LangChain's `LLMChain`.

4. The model responds with a styled outfit recommendation written in fluent, fashion-aware English.

5. The app displays this response on the frontend, styled using Streamlit’s custom theming.

---

## 🎯 Purpose

This project showcases how Large Language Models can be applied to real-world use cases like fashion personalization. It also demonstrates the integration of AI with interactive web apps, making it a great example of applied NLP and UI/UX.

---

## ✅ Status

The application is complete and successfully deployed as a demo. All features are tested and functional.

---

## 🔗 Connect with Me

- 📧 Email: ashadakber32@gmail.com  
- 💼 [LinkedIn](https://www.linkedin.com/in/ashad-k)  
- 🐙 [GitHub](https://github.com/Ashad777)
