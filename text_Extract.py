import PyPDF2

# 1. Open the PDF file
pdf_file = open("life-30.pdf", "rb")   # replace with your PDF file name

# 2. Create PDF reader
reader = PyPDF2.PdfReader(pdf_file)

# 3. Create/clear text file
with open("extracted_text.txt", "w", encoding="utf-8") as text_file:
    # 4. Loop through all pages and extract text
    for page_num in range(len(reader.pages)):
        text = reader.pages[page_num].extract_text()
        if text:
            text_file.write(text + "\n")

pdf_file.close()

print("✅ Text extracted and saved in extracted_text.txt")
