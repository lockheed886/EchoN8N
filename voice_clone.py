import os
import re

# 🔹 Must be set before importing TTS
os.environ["COQUI_TOS_AGREED"] = "1"  # auto accept license
os.environ["COQUI_TTS_CACHE"] = os.path.expanduser("~/Downloads/Voice Clone/coqui_cache")  # cache directory

# ====================================================================
# FINAL FIX FOR THE PYTORCH ERROR
# We are adding all four required classes to PyTorch's safe list.
import torch.serialization
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
# ====================================================================

from TTS.api import TTS
from pydub import AudioSegment

def list_cached_models():
    """List all cached TTS models"""
    cache_dirs = [
        os.path.expanduser("~/.local/share/tts"),
        os.path.expanduser("~/Downloads/Voice Clone/coqui_cache")
    ]
    
    cached_models = []
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            for item in os.listdir(cache_dir):
                item_path = os.path.join(cache_dir, item)
                if os.path.isdir(item_path) and len(os.listdir(item_path)) > 0:
                    # Convert directory name back to model name
                    model_name = item.replace("--", "/")
                    if model_name not in cached_models:
                        cached_models.append(model_name)
    
    return cached_models

def check_model_cached(model_name):
    """Check if a model is already cached locally"""
    # Check in default TTS cache directory
    default_cache = os.path.expanduser("~/.local/share/tts")
    
    # Convert model name to directory format
    model_dir_name = model_name.replace("/", "--")
    model_path = os.path.join(default_cache, model_dir_name)
    
    # Also check our custom cache directory
    custom_cache = os.path.expanduser("~/Downloads/Voice Clone/coqui_cache")
    custom_model_path = os.path.join(custom_cache, model_dir_name)
    
    # Check if model exists in either location and has essential files
    for path in [model_path, custom_model_path]:
        if os.path.exists(path) and os.path.isdir(path):
            # Check for essential model files
            files = os.listdir(path)
            if len(files) > 0:  # If directory has any files, consider it cached
                print(f"✅ Found cached model at: {path}")
                return True
    
    print(f"❌ Model {model_name} not found in cache")
    return False

def main():
    # Show available cached models first
    print("🔍 Checking for cached TTS models...")
    cached_models = list_cached_models()
    if cached_models:
        print(f"✅ Found {len(cached_models)} cached model(s):")
        for model in cached_models:
            print(f"   - {model}")
    else:
        print("❌ No cached models found. Will need to download a model.")
    
    print("\n" + "="*60)
    print("Loading TTS model for voice cloning...")
    
    # For voice cloning, we need XTTS model specifically
    print("🎯 Loading XTTS model for voice cloning...")
    print("📝 XTTS model is required for voice cloning functionality")
    
    tts = None
    model_used = None
    
    # Try to load XTTS model first (required for voice cloning)
    xtts_model = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    try:
        print(f"Loading XTTS model: {xtts_model}")
        tts = TTS(model_name=xtts_model, progress_bar=True, gpu=False)
        model_used = xtts_model
        print(f"✅ Successfully loaded XTTS model for voice cloning")
    except Exception as e:
        print(f"❌ Failed to load XTTS model: {str(e)[:200]}...")
        print("⚠️  XTTS model failed. Trying fallback models without voice cloning...")
        
        # Use working fallback models if XTTS fails
        fallback_models = [
            "tts_models/en/ljspeech/glow-tts",
            "tts_models/en/ljspeech/speedy-speech",
            "tts_models/en/ljspeech/tacotron2-DDC"
        ]
        
        for model_name in fallback_models:
            try:
                print(f"Loading fallback model: {model_name}")
                tts = TTS(model_name=model_name, progress_bar=False, gpu=False)
                model_used = model_name
                print(f"✅ Successfully loaded fallback model: {model_name}")
                print("ℹ️  This model will use its default voice (voice cloning not available)")
                break
            except Exception as e2:
                print(f"❌ Failed to load {model_name}: {str(e2)[:100]}...")
                continue
    
    if tts is None:
        print("❌ All models failed to load. Please check your internet connection and try again later.")
        return
        
    print(f"Using model: {model_used}")

    # 🔹 Path to your voice sample (~10 sec clear audio)
    speaker_wav = "myvoice.wav"
    voice_cloning_available = "xtts" in model_used.lower() and os.path.exists(speaker_wav)
    
    print(f"\n🎭 Voice Cloning Setup:")
    print(f"   - Model: {model_used}")
    print(f"   - Voice sample: {speaker_wav}")
    
    if not os.path.exists(speaker_wav):
        print(f"❌ Voice sample file NOT found: {speaker_wav}")
        if "xtts" in model_used.lower():
            print("⚠️  XTTS model requires a voice sample for cloning!")
            print("📝 Please ensure 'myvoice.wav' exists for voice cloning to work.")
            print("🔄 Continuing with XTTS default voice...")
        else:
            print("ℹ️  Using model's default voice.")
    else:
        file_size = os.path.getsize(speaker_wav)
        print(f"✅ Voice sample found: {speaker_wav} ({file_size} bytes)")
        if voice_cloning_available:
            print("🎯 Voice cloning ENABLED - will clone your voice!")
        else:
            print("⚠️  Voice cloning NOT available with current model")
    
    print(f"   - Voice cloning status: {'✅ ENABLED' if voice_cloning_available else '❌ DISABLED'}")
    print()

    # 🔹 Read text from file
    try:
        with open("extractedText.txt", "r", encoding="utf-8") as f:
            input_text = f.read()
    except FileNotFoundError:
        print("Error: extractedText.txt not found. Please create this file and add text to it.")
        return

    # 1. Clean the text
    print("Cleaning input text...")
    clean_text = input_text.replace('\t', ' ').replace('\n', ' ').strip()
    clean_text = re.sub(r'\s+', ' ', clean_text)

    # 2. Split the text into sentences and limit for testing
    sentences = re.split(r'(?<=[.!?])\s+', clean_text)
    
    # Limit to first 10 sentences for testing to avoid overwhelming the system
    if len(sentences) > 10:
        print(f"⚠️  Large text detected ({len(sentences)} sentences). Using first 10 sentences for testing.")
        sentences = sentences[:10]
    
    temp_audio_files = []
    print(f"Found {len(sentences)} sentences to synthesize.")

    # 3. Generate audio for each sentence with voice cloning
    successful_files = 0
    voice_cloned_files = 0
    
    for i, sentence in enumerate(sentences):
        if not sentence.strip():
            continue
        
        temp_file_path = f"temp_audio_{i}.wav"
        print(f"[{i+1}/{len(sentences)}] Synthesizing: {sentence[:80]}...")
        
        synthesis_successful = False
        used_voice_cloning = False
        
        # Try voice cloning if XTTS model and voice sample are available
        if voice_cloning_available:
            try:
                print(f"  🎯 Using voice cloning with {speaker_wav}...")
                tts.tts_to_file(
                    text=sentence,
                    speaker_wav=speaker_wav,
                    language="en",
                    file_path=temp_file_path
                )
                temp_audio_files.append(temp_file_path)
                synthesis_successful = True
                used_voice_cloning = True
                successful_files += 1
                voice_cloned_files += 1
                print(f"  ✅ Voice cloning successful! (cloned voice)")
            except Exception as e:
                print(f"  ❌ Voice cloning failed: {str(e)[:200]}...")
                print(f"  🔄 Trying XTTS with default voice...")
                
                # Try XTTS with default voice as fallback
                try:
                    # For XTTS, we can try with a built-in speaker
                    tts.tts_to_file(
                        text=sentence,
                        file_path=temp_file_path,
                        language="en"
                    )
                    temp_audio_files.append(temp_file_path)
                    synthesis_successful = True
                    successful_files += 1
                    print(f"  ✅ XTTS synthesis with default voice successful")
                except Exception as e2:
                    print(f"  ❌ XTTS default voice also failed: {str(e2)[:150]}...")
        
        # Try standard synthesis if XTTS failed or not available
        if not synthesis_successful:
            try:
                print(f"  🔄 Using standard synthesis...")
                tts.tts_to_file(
                    text=sentence,
                    file_path=temp_file_path
                )
                temp_audio_files.append(temp_file_path)
                synthesis_successful = True
                successful_files += 1
                print(f"  ✅ Standard synthesis successful")
            except Exception as e:
                print(f"  ❌ Standard synthesis failed: {str(e)[:150]}...")
        
        # If all approaches failed, skip this sentence
        if not synthesis_successful:
            print(f"  ⚠️  Skipping sentence {i+1} - all synthesis methods failed")
    
    print(f"\n📊 Synthesis Results:")
    print(f"   - Total sentences: {len(sentences)}")
    print(f"   - Successful: {successful_files}")
    print(f"   - Voice cloned: {voice_cloned_files}")
    print(f"   - Failed: {len(sentences) - successful_files}")
    
    if voice_cloned_files > 0:
        print(f"🎉 Successfully cloned your voice for {voice_cloned_files} sentences!")
    elif voice_cloning_available and successful_files > 0:
        print(f"🔊 Used XTTS default voice (voice cloning had issues)")
    elif successful_files > 0:
        print(f"🔊 Used standard model voice")
    
    print(f"� Success rate: {(successful_files/len(sentences)*100):.1f}%")

    # 4. Combine all temporary audio files
    if not temp_audio_files:
        print("\n❌ No audio files were generated successfully.")
        print("This might be due to:")
        print("   - Model compatibility issues")
        print("   - Missing or incompatible speaker voice file")
        print("   - Text formatting problems")
        print("   - Network/dependency issues")
        return
    
    print(f"\n🔄 Combining {len(temp_audio_files)} audio clips...")
    combined_audio = AudioSegment.empty()
    combined_count = 0
    
    for file_path in temp_audio_files:
        try:
            if os.path.exists(file_path):
                segment = AudioSegment.from_wav(file_path)
                combined_audio += segment
                combined_count += 1
                print(f"  ✅ Added {os.path.basename(file_path)}")
            else:
                print(f"  ❌ File not found: {file_path}")
        except Exception as e:
            print(f"  ❌ Could not process {file_path}: {e}")
        finally:
            # 5. Clean up temporary files
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass  # Ignore cleanup errors

    if combined_count == 0:
        print("\n❌ No audio clips could be combined.")
        return

    # Export the final audio
    final_output_path = "cloned_voice.wav"
    try:
        combined_audio.export(final_output_path, format="wav")
        print(f"\n✅ Audio synthesis complete!")
        print(f"📁 Output saved as: {final_output_path}")
        print(f"📊 Successfully combined {combined_count} audio segments")
        
        # Show file info
        duration_seconds = len(combined_audio) / 1000.0
        print(f"🎵 Duration: {duration_seconds:.1f} seconds")
        
        if voice_cloning_available:
            print(f"🎭 The audio uses your cloned voice from: {speaker_wav}")
        else:
            print(f"🔊 The audio uses the default model voice")
            if os.path.exists(speaker_wav):
                print(f"💡 Tip: To use voice cloning, ensure you have XTTS model working properly")
        
    except Exception as e:
        print(f"\n❌ Failed to export final audio: {e}")
        return

if __name__ == "__main__":
    main()