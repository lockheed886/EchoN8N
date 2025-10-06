from gtts import gTTS


# 1. Open the text file
with open("extracted_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 2. Convert text to speech
tts = gTTS(text=text, lang='en', slow=False)

# 3. Save as audio file
tts.save("output_audio.mp3")

print("✅ Audio saved as output_audio.mp3")
