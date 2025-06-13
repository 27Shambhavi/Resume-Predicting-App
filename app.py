# You need to install these dependencies before running:
# pip install streamlit scikit-learn python-docx PyPDF2

import streamlit as st
import pickle
import docx  # for DOCX extraction
import PyPDF2  # for PDF extraction
import re

# Load trained model and TF-IDF vectorizer
clf = pickle.load(open('clf.pkl', 'rb'))      # Trained classifier
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Trained TF-IDF vectorizer

# Category mapping (manual class name mapping)
category_mapping = {
     6: 'Data Science', 
    12: 'HR', 
    0: 'Advocate', 
    1: 'Arts', 
    24: 'Web Designing',
    16: 'Mechanical Engineer', 
    22: 'Sales', 
    14: 'Health and fitness', 
    5: 'Civil Engineer',
    15: 'Java Developer', 
    4: 'Business Analyst', 
    21: 'SAP Developer', 
    2: 'Automation Testing',
    11: 'Electrical Engineering', 
    18: 'Operations Manager', 
    20: 'Python Developer',
    8: 'DevOps Engineer', 
    17: 'Network Security Engineer', 
    19: 'PMO', 
    7: 'Database',
    13: 'Hadoop', 
    10: 'ETL Developer', 
    9: 'DotNet Developer', 
    3: 'Blockchain', 
    23: 'Testing'
}

# Function to clean text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText


# Extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Handle file upload and text extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

# Predict category
def predict_category(resume_text):
    cleaned_text = cleanResume(resume_text)
    vectorized_text = tfidf.transform([cleaned_text])  # Do not convert to array
    predicted_id = clf.predict(vectorized_text)[0]
    return category_mapping.get(predicted_id, "Unknown")

# Streamlit UI
def main():
    st.set_page_config(page_title="Resume Category Predictor", layout="wide", page_icon="📄")
    st.title("📄 Resume Category Predictor")
    st.markdown("Upload a resume (PDF, DOCX, or TXT) to predict the most likely job category.")

    uploaded_file = st.file_uploader("Upload Resume File", type=["pdf", "docx", "txt"])

    if uploaded_file:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Resume text extracted successfully.")

            if st.checkbox("Show Extracted Resume Text"):
                st.text_area("Extracted Text", resume_text, height=300)

            predicted_category = predict_category(resume_text)
            st.subheader("🧠 Predicted Job Category:")
            st.success(f"🎯 **{predicted_category}**")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
