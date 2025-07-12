import google.generativeai as genai

genai.configure(api_key="AIzaSyA_eprNsgB-RtxuGMhIlsbwJnHdMHhyRME")

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction=(
        "You are an agricultural expert and farming advisor. "
        "Provide accurate, location-based, season-aware, and practical farming advice. "
        "Reply in friendly and helpful tone. Prefer Hindi if question is in Hindi."
    )
)

chat = model.start_chat()

print("🌾 Welcome to FarmMitra Chatbot (Powered by Gemini AI)")
print("💬 Ask any farming-related question.")
print("📌 Type 'exit' to quit.\n")

while True:
    user_input = input("👨‍🌾 You: ")
    if user_input.lower() in ["exit", "quit", "bye"]:
        print("👋 Thank you for using FarmMitra! Happy Farming! 🙏")
        break
    try:
        response = chat.send_message(user_input)
        print("🤖 Gemini:", response.text.strip(), "\n")
    except Exception as e:
        print("⚠️ Error:", str(e), "\n")









