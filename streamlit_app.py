import streamlit as st
import PyPDF2
from gtts import gTTS
import os
import re
import io
from pydub import AudioSegment

# 🔹 Must be set before importing TTS
os.environ["COQUI_TOS_AGREED"] = "1"
os.environ["COQUI_TTS_CACHE"] = os.path.expanduser("~/Downloads/Voice Clone/coqui_cache")

# Import TTS after setting environment variables
# Import torch serialization and add safe globals lazily to avoid circular imports
def setup_torch_safe_globals():
    """Setup torch serialization safe globals - called only when needed"""
    try:
        import torch.serialization
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
        from TTS.config.shared_configs import BaseDatasetConfig
        
        torch.serialization.add_safe_globals([
            XttsConfig, 
            XttsAudioConfig, 
            BaseDatasetConfig, 
            XttsArgs
        ])
        return True
    except Exception as e:
        st.warning(f"Could not setup torch safe globals: {e}")
        return False

# Page configuration
st.set_page_config(
    page_title="PDF to Voice Clone",
    page_icon="🎙️",
    layout="wide"
)

# Title and description
st.title("🎙️ PDF to Voice Clone Application")
st.markdown("""
This application extracts text from PDF files and generates audio using:
1. **Basic Text-to-Speech** (gTTS) - Quick and simple
2. **Voice Cloning** (Coqui TTS) - Clone your voice!
""")

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'gtts_audio' not in st.session_state:
    st.session_state.gtts_audio = None
if 'cloned_audio' not in st.session_state:
    st.session_state.cloned_audio = None

# ==================== FUNCTION DEFINITIONS ====================

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file with robust error handling"""
    
    # Try PyPDF2 first (faster)
    try:
        # Reset file pointer to beginning
        pdf_file.seek(0)
        
        # Try with strict=False to handle malformed PDFs
        pdf_reader = PyPDF2.PdfReader(pdf_file, strict=False)
        
        # Extract text from all pages
        extracted_text = ""
        total_pages = len(pdf_reader.pages)
        
        if total_pages == 0:
            raise Exception("PDF appears to be empty")
        
        st.info(f"📄 Processing {total_pages} page(s) with PyPDF2...")
        
        for page_num in range(total_pages):
            try:
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text and text.strip():
                    extracted_text += text + "\n"
            except Exception as page_error:
                st.warning(f"⚠️ Could not extract text from page {page_num + 1}: {str(page_error)}")
                continue
        
        if extracted_text.strip():
            st.success(f"✅ Successfully extracted text using PyPDF2!")
            return extracted_text.strip()
        else:
            # If no text extracted, try fallback method
            raise Exception("No text extracted with PyPDF2")
            
    except Exception as pypdf_error:
        st.warning(f"⚠️ PyPDF2 failed: {str(pypdf_error)}")
        st.info("🔄 Trying alternative method with pdfplumber...")
        
        # Try pdfplumber as fallback
        try:
            import pdfplumber
            
            # Reset file pointer
            pdf_file.seek(0)
            
            extracted_text = ""
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                
                if total_pages == 0:
                    st.error("❌ PDF appears to be empty or has no readable pages.")
                    return None
                
                st.info(f"📄 Processing {total_pages} page(s) with pdfplumber...")
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            extracted_text += text + "\n"
                    except Exception as page_error:
                        st.warning(f"⚠️ Could not extract text from page {page_num + 1}: {str(page_error)}")
                        continue
            
            if extracted_text.strip():
                st.success(f"✅ Successfully extracted text using pdfplumber!")
                return extracted_text.strip()
            else:
                st.error("❌ No text could be extracted. The PDF might be image-based or encrypted.")
                st.info("💡 Tips:")
                st.markdown("""
                - If your PDF contains scanned images, you'll need OCR (Optical Character Recognition)
                - Try converting the PDF to text format first
                - Ensure the PDF is not password-protected
                """)
                return None
                
        except ImportError:
            st.error("❌ pdfplumber is not installed. Please install it: `pip install pdfplumber`")
            return None
        except Exception as plumber_error:
            st.error(f"❌ Both PDF extraction methods failed.")
            st.error(f"Final error: {str(plumber_error)}")
            st.info("💡 Suggestions:")
            st.markdown("""
            - Re-save the PDF using a PDF editor (Adobe Acrobat, Preview, etc.)
            - Convert to a standard PDF format
            - Try a different PDF file
            - Check if the PDF requires a password
            """)
            return None

def generate_gtts_audio(text):
    """Generate audio using gTTS"""
    try:
        # Create gTTS object
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to bytes
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        return audio_bytes.read()
    except Exception as e:
        st.error(f"Error generating gTTS audio: {str(e)}")
        return None

def load_tts_model():
    """Load TTS model for voice cloning"""
    try:
        # Import TTS here to avoid circular import issues
        from TTS.api import TTS
        
        # Setup torch safe globals for XTTS model
        setup_torch_safe_globals()
        
        xtts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
        tts = TTS(model_name=xtts_model, progress_bar=True, gpu=False)
        return tts, xtts_model
    except Exception as e:
        st.warning(f"Failed to load XTTS model: {str(e)[:200]}. Trying fallback models...")
        
        # Use working fallback models if XTTS fails
        fallback_models = [
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/ljspeech/speedy-speech",
            "tts_models/en/ljspeech/tacotron2-DDC"
        ]
        
        for model_name in fallback_models:
            try:
                from TTS.api import TTS
                tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
                st.info(f"Using fallback model: {model_name} (voice cloning not available)")
                return tts, model_name
            except Exception as e2:
                continue
        
        return None, None

def generate_cloned_audio(text, speaker_wav_file, tts_model):
    """Generate audio with voice cloning"""
    try:
        # Clean the text
        clean_text = text.replace('\t', ' ').replace('\n', ' ').strip()
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        
        # Limit to first 10 sentences for testing
        if len(sentences) > 10:
            st.warning(f"Large text detected ({len(sentences)} sentences). Using first 10 sentences.")
            sentences = sentences[:10]
        
        # Save speaker wav temporarily
        temp_speaker_path = "temp_speaker.wav"
        with open(temp_speaker_path, "wb") as f:
            f.write(speaker_wav_file.read())
        
        temp_audio_files = []
        tts, model_name = tts_model
        voice_cloning_available = "xtts" in model_name.lower()
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        successful_files = 0
        voice_cloned_files = 0
        
        # Generate audio for each sentence
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            temp_file_path = f"temp_audio_{i}.wav"
            status_text.text(f"Processing sentence {i+1}/{len(sentences)}...")
            
            synthesis_successful = False
            
            # Try voice cloning if XTTS model is available
            if voice_cloning_available:
                try:
                    tts.tts_to_file(
                        text=sentence,
                        speaker_wav=temp_speaker_path,
                        language="en",
                        file_path=temp_file_path
                    )
                    temp_audio_files.append(temp_file_path)
                    synthesis_successful = True
                    successful_files += 1
                    voice_cloned_files += 1
                except Exception as e:
                    # Try XTTS with default voice as fallback
                    try:
                        tts.tts_to_file(
                            text=sentence,
                            file_path=temp_file_path,
                            language="en"
                        )
                        temp_audio_files.append(temp_file_path)
                        synthesis_successful = True
                        successful_files += 1
                    except Exception as e2:
                        pass
            
            # Try standard synthesis if XTTS failed or not available
            if not synthesis_successful:
                try:
                    tts.tts_to_file(
                        text=sentence,
                        file_path=temp_file_path
                    )
                    temp_audio_files.append(temp_file_path)
                    synthesis_successful = True
                    successful_files += 1
                except Exception as e:
                    pass
            
            progress_bar.progress((i + 1) / len(sentences))
        
        status_text.text(f"Combining {len(temp_audio_files)} audio clips...")
        
        # Combine all audio files
        if not temp_audio_files:
            st.error("No audio files were generated successfully.")
            return None
        
        combined_audio = AudioSegment.empty()
        for file_path in temp_audio_files:
            try:
                if os.path.exists(file_path):
                    segment = AudioSegment.from_wav(file_path)
                    combined_audio += segment
            except Exception as e:
                pass
            finally:
                # Clean up temporary files
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except:
                    pass
        
        # Export to bytes
        output_buffer = io.BytesIO()
        combined_audio.export(output_buffer, format="wav")
        output_buffer.seek(0)
        
        # Clean up temporary speaker file
        try:
            if os.path.exists(temp_speaker_path):
                os.remove(temp_speaker_path)
        except:
            pass
        
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        if voice_cloned_files > 0:
            st.success(f"✅ Successfully cloned your voice for {voice_cloned_files} sentences!")
        elif successful_files > 0:
            st.info(f"🔊 Generated audio for {successful_files} sentences (using model default voice)")
        
        return output_buffer.read()
        
    except Exception as e:
        st.error(f"Error generating cloned audio: {str(e)}")
        return None

# ==================== STREAMLIT UI ====================

# Step 1: Upload PDF
st.header("📄 Step 1: Upload PDF File")
pdf_file = st.file_uploader("Choose a PDF file", type=['pdf'])

if pdf_file is not None:
    if st.button("Extract Text from PDF"):
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(pdf_file)
            if extracted_text:
                st.session_state.extracted_text = extracted_text
                st.success("✅ Text extracted successfully!")

# Display extracted text
if st.session_state.extracted_text:
    st.subheader("Extracted Text:")
    with st.expander("View/Edit Extracted Text", expanded=False):
        st.session_state.extracted_text = st.text_area(
            "You can edit the text before generating audio:",
            st.session_state.extracted_text,
            height=300
        )
    
    st.info(f"Text length: {len(st.session_state.extracted_text)} characters")
    
    # Step 2: Generate Basic TTS
    st.header("🔊 Step 2: Generate Basic Text-to-Speech")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate gTTS Audio", use_container_width=True):
            with st.spinner("Generating audio with gTTS..."):
                audio_bytes = generate_gtts_audio(st.session_state.extracted_text)
                if audio_bytes:
                    st.session_state.gtts_audio = audio_bytes
                    st.success("✅ gTTS audio generated!")
    
    with col2:
        if st.session_state.gtts_audio:
            st.audio(st.session_state.gtts_audio, format='audio/mp3')
            st.download_button(
                label="⬇️ Download gTTS Audio",
                data=st.session_state.gtts_audio,
                file_name="output_audio.mp3",
                mime="audio/mp3",
                use_container_width=True
            )
    
    # Step 3: Voice Cloning
    st.header("🎙️ Step 3: Generate Voice Cloned Audio")
    st.markdown("""
    Upload a sample of your voice (WAV format, ~10 seconds of clear audio) to clone it.
    """)
    
    voice_sample = st.file_uploader("Upload Voice Sample (WAV file)", type=['wav'])
    
    if voice_sample is not None:
        st.audio(voice_sample, format='audio/wav')
        
        if st.button("Generate Voice Cloned Audio", use_container_width=True):
            with st.spinner("Loading TTS model... This may take a while on first run."):
                tts_model = load_tts_model()
                
                if tts_model[0] is None:
                    st.error("❌ Failed to load any TTS model. Please check your installation.")
                else:
                    with st.spinner("Generating voice cloned audio... This may take several minutes."):
                        cloned_audio = generate_cloned_audio(
                            st.session_state.extracted_text,
                            voice_sample,
                            tts_model
                        )
                        if cloned_audio:
                            st.session_state.cloned_audio = cloned_audio
                            st.success("✅ Voice cloned audio generated!")
    
    # Display cloned audio
    if st.session_state.cloned_audio:
        st.subheader("Voice Cloned Audio:")
        col1, col2 = st.columns(2)
        
        with col1:
            st.audio(st.session_state.cloned_audio, format='audio/wav')
        
        with col2:
            st.download_button(
                label="⬇️ Download Voice Cloned Audio",
                data=st.session_state.cloned_audio,
                file_name="cloned_voice.wav",
                mime="audio/wav",
                use_container_width=True
            )

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    ### How to use:
    
    1. **Upload PDF**: Select a PDF file to extract text from
    2. **Extract Text**: Click to extract text from the PDF
    3. **Generate gTTS Audio**: Create basic text-to-speech audio
    4. **Upload Voice Sample**: Upload a WAV file with your voice (~10 seconds)
    5. **Generate Voice Clone**: Create audio that sounds like you!
    
    ### Tips:
    - For best voice cloning results, use a clear voice sample
    - Voice sample should be ~10 seconds long
    - First run may take time to download models
    - Large texts are limited to 10 sentences for faster processing
    
    ### Audio Options:
    - ✅ Play audio directly in the browser
    - ⬇️ Download audio files for offline use
    """)
    
    st.header("ℹ️ About")
    st.markdown("""
    **Technologies:**
    - PyPDF2 for PDF text extraction
    - gTTS for basic text-to-speech
    - Coqui TTS for voice cloning
    - Streamlit for UI
    """)
