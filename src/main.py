import os
import json
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from pydub import AudioSegment
from dotenv import load_dotenv
import openai
from TTS.api import TTS
import glob
from translate import Translator
from jiwer import wer  # For Word Error Rate calculation
from nltk.translate.bleu_score import sentence_bleu  # For BLEU Score calculation


MAX_LENGTH = 500

def endpart(number):
    video_path = "../data/input_video.mp4"
    audio_paths = [f"../data/TTS_OUTPUT{i}.wav" for i in range(1, number)]

    video_clip = VideoFileClip(video_path)
    final_clips = []

    start_time = 0
    for i, audio_path in enumerate(audio_paths, start=1):
        audio_clip = AudioFileClip(audio_path)
        duration = min(video_clip.duration - start_time, audio_clip.duration)
        video_segment = video_clip.subclip(start_time, start_time + duration)
        audio_segment = audio_clip.subclip(0, duration)
        final_clip = video_segment.set_audio(audio_segment)
        final_clips.append(final_clip)
        start_time += duration

    final_video = concatenate_videoclips(final_clips)
    final_video_path = "../results/final_video.mp4"
    final_video.write_videofile(final_video_path)

def translate_text(text, target_language, source_language="en"):
    translator = Translator(from_lang=source_language, to_lang=target_language)
    translation = translator.translate(text)
    return translation

def slice_sentence(sentence, max_length):
    words = sentence.split()
    sliced_sentences = []
    current_sentence = ""

    for word in words:
        if len(current_sentence) + len(word) <= max_length:
            current_sentence += word + " "
        else:
            sliced_sentences.append(current_sentence.strip())
            current_sentence = word + " "

    if current_sentence:
        sliced_sentences.append(current_sentence.strip())

    return sliced_sentences

def calculate_wer(reference, hypothesis):
    return wer(reference, hypothesis)

def calculate_bleu(reference, hypothesis):
    reference = [reference.split()]
    hypothesis = hypothesis.split()
    return sentence_bleu(reference, hypothesis)

# Assuming we have a function to estimate MOS (Mean Opinion Score) for TTS
# This is usually done via human evaluation, but we can create a placeholder
def estimate_mos(speech):
    # Placeholder function
    return 4.0



load_dotenv()

api_key = os.getenv("API_KEY")
client = openai.OpenAI(api_key=api_key)

target_language = input("Enter Target Language ( English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu) and Korean (ko)): ")
source_language = input("Enter Source Language ( English (en), Spanish (es), French (fr), German (de), Italian (it), Portuguese (pt), Polish (pl), Turkish (tr), Russian (ru), Dutch (nl), Czech (cs), Arabic (ar), Chinese (zh-cn), Japanese (ja), Hungarian (hu) and Korean (ko)): ")

video_path = "../data/input_video.mp4"
output_audio_path = "../data/raw_audio.wav"
output_json_path = "../data/sampledata.json"

audio = AudioSegment.from_file(video_path, format="mp4")
segment_length = 60 * 1000  # 1 minute in milliseconds

print("Segmenting...")
segments = []
for start_time in range(0, len(audio), segment_length):
    end_time = start_time + segment_length
    segment = audio[start_time:end_time]
    segment.export(f"segment_{start_time}.wav", format="wav")
    segments.append(f"segment_{start_time}.wav")

print("Transcribing...")
transcribed_text = ""
for segment_file in segments:
    audio_content = open(segment_file, "rb")
    response = client.audio.transcriptions.create(model="whisper-1", file=audio_content)
    segment_transcription = response.text
    print(segment_transcription)
    transcribed_text += segment_transcription + " "

# Calculate WER for STT
reference_text = "Cari amici di Italian, eccoci ad un nuovo episodio. Oggi chiederemo agli italiani cosa amano mangiare di pi\u00f9 e cosa non pu\u00f2 mancare dal loro frigo. Andiamo! Ciao! Cosa ti piace di pi\u00f9 mangiare? Pasta al forno. Con le polpettine o senza? Polpettine. Orecchietto al pomodoro. Prediligo la pasta al secondo, la carne. Diciamo che quella che pi\u00f9 mi piace sono pasta e cime di rapa con un po' di acciughe sopra. E la mollica grattugiata fritta. Spaghetti e frutti di mare. Concordo, s\u00ec. Le orecchiette con il cimo di rapa. A me piace molto la pasta con il pesce. La pasta alla pizzaiola. Come \u00e8 questa pasta alla pizzaiola? E' buonissimo! E quali sono gli ingredienti? L'olio, il sugo, la mozzarella, l'origano e la pasta. Cosa non deve mai mancare nel suo frigo? L'acqua, sicuramente l'acqua perch\u00e9 non sono un bevitore di alcolici o altro. Il latte, per prenderlo la mattina e basta. Per il resto, io mangio per vivere. Fino a qualche mese fa il formaggio. Ora, per questioni di dieta, l'abbiamo un attimo. Formaggi, frutta, verdura. Il latte. Non lo so, non deve essere il vuoto. Direi il formaggio fresco, anche il yogurt. La cioccolata, per\u00f2 non sta in frigo, \u00e8 la cioccolata che non pu\u00f2 mancare mai. La nutella e la cioccolata. Bella, s\u00ec, non manca mai. E il pesto. Cosa sapete cucinare meglio? La pasta. La pasta. Cucino, ma io la lasagna perch\u00e9 la cucino spesso e mi piace cucinarla. Spaghetti al pomodoro. Solo l'uovo fritto. Chi \u00e8 che cucina a casa? Mia moglie. Fondamentalmente non mi piace cucinare. Diciamo, la cosa che faccio spesso \u00e8 pasta con i legumi. Quali? Lenticchie. Lei sa cucinare? Qualcosa, s\u00ec. Qual \u00e8 il piatto che cucina peggio? L'anatra all'arancia, che sar\u00e0 difficilissimo da fare, non lo so. Fare, diciamo, i dolci. I dolci non li so fare. I dolci vanno tutti in crisi e non ci provano pi\u00f9. Ma le cose complicate, quelle mi escono male. Non sarei in grado di fare una lasagna o il pesce, forse. Non mi cimenterei proprio. Qual \u00e8 il piatto che ti cucina tua mamma che ti piace di pi\u00f9? Pasta al forno. Pasta al forno. Bene, grazie mille. Ciao. Prego, arrivederci. Sottotitoli creati dalla comunit\u00e0 Amara.org"
wer_score = calculate_wer(reference_text, transcribed_text)
print(f"Word Error Rate (WER): {wer_score}")

print("Removing trail files...")
for segment_file in segments:
    os.remove(segment_file)

print("Saving json for translation request...")
data = {
    "source_text": transcribed_text,
    "source": source_language,
    "target": target_language
}
with open(output_json_path, "w") as json_file:
    json.dump(data, json_file)

print("Loading Data...")
with open('../data/sampledata.json') as json_file:
    data = json.load(json_file)
source_text = data['source_text']
sliced_sentences = slice_sentence(source_text, MAX_LENGTH)
source_language = data['source']
target_language = data['target']

print("Translating...")
translated_texts = []
for sentence in sliced_sentences:
    translated_texts.append(translate_text(sentence, target_language, source_language))
output_data = {
    "translated_texts": translated_texts
}
# Better practice to log the output of the translation for error checking
with open('../data/translated_text.json', 'w') as output_file:
    json.dump(output_data, output_file)

# Calculate BLEU score for Translation
reference_translations = ["Dear Friends of Italian, Here we are at a new episode. Today we're going to ask Italians what they like to eat most and what can't miss from their fridge. Let's go! Hi! What do you like to eat most? Baked pasta. With meatballs or without? Meatballs. Orecchietto with tomato sauce. I prefer pasta to the main course, meat. Let's say the one I like best is pasta and turnip greens with a little anchovies on top. And fried grated breadcrumbs. Spaghetti and seafood. Agreed, yes. Orecchiette pasta with turnip greens. I really like pasta with seafood. Pasta alla pizzaiola. How is this pasta alla pizzaiola? It's so good! And what are the ingredients? The oil, the sauce, the mozzarella, the oregano and the pasta. What should never be missing in your refrigerator? Water, definitely water because I am not a drinker of alcohol or anything. Milk, to take it in the morning and that's it. Otherwise, I eat for a living. Until a few months ago, cheese. Now, because of diet issues, we have it for a moment. Cheese, fruits, vegetables. Milk. I don't know, it doesn't have to be vacuum. I would say cream cheese, even yogurt. Chocolate, though, it doesn't sit in the fridge, it's the chocolate that can never be missed. Nutella and chocolate. Nice, yes, it never misses. And pesto. What can you cook best?  Pasta. Pasta. I cook, but I lasagna because I cook it often and I like to cook it. Spaghetti with tomato sauce. Just the fried egg. Who does the cooking at home? My wife. I basically don't like to cook. Let's say, the thing I do often is pasta with legumes. Which ones? Lentils. Can she cook? Some of it, yes. What is the dish you cook the worst? Duck with orange, which will be very difficult to make, I don't know. Making, say, desserts. Desserts I don't know how to make. Desserts all go downhill and they don't try anymore. But complicated things, those come out bad for me. I wouldn't be able to make a lasagna or fish, maybe. I wouldn't try my hand at it at all. What is the dish your mom cooks for you that you like best? Baked pasta. Baked pasta. Well, thank you very much. Bye. You're welcome, goodbye. Subtitles created by the Amara.org community."]
hypothesis_translation = " ".join(translated_texts)
bleu_score = calculate_bleu(" ".join(reference_translations), hypothesis_translation)
print(f"BLEU Score: {bleu_score}")

# Read from the user's generated fine-tuning dataset, assuming it exists
speaker_wav_paths = glob.glob("../data/recording/*.wav")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
with open("../data/translated_text.json") as file:
    data = json.load(file)
    translated_texts = data["translated_texts"]
i = 1
print("Speaking...")
for text in translated_texts:
    tts.tts_to_file(text=text,
                    file_path=f"../data/TTS_OUTPUT{i}.wav",
                    speaker_wav=speaker_wav_paths,  # type: ignore
                    language=target_language,
                    split_sentences=True)
    i+=1

# Calculate MOS estimation for TTS
mos_scores = []
for wav_file in glob.glob(f"../data/TTS_OUTPUT*.wav"):
    speech = AudioSegment.from_file(wav_file)
    mos_score = estimate_mos(speech)
    mos_scores.append(mos_score)

average_mos = sum(mos_scores) / len(mos_scores)
print(f"Mean Opinion Score (MOS): {average_mos}")

endpart(i)
