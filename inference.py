from tensorflow import keras
import tensorflow as tf 
import re 
# load model
model = keras.models.load_model('models/bilstm_model.keras')
print('\nBI-LSTM model is loaded successfully. \n')
# Basic text cleaning function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Inference Test Function
def predict_comment_sentiment(comment_text):
    print('Result: ')
    cleaned = preprocess_text(comment_text)
    input_data = tf.constant([cleaned], dtype=tf.string)
    prob = model.predict(input_data, verbose=1)[0][0]
    predicted_class = 1 if prob >= 0.5 else 0
    class_label = "Pro" if predicted_class == 1 else "Con / Skeptical / OOT"
    print(f"[Probability Pro]: {prob:.4f}")
    print(f"[Prediction]: Class {predicted_class} -> {class_label}")

# main program
def main():
    print('\n === SENTIMENT PREDICTOR TO DETECT DOOM SCROLLERS FROM THEIR COMMENTS ON SOCIAL MEDIA === \n')
    prompt = input('Input comment/prompt (in Indonesian [ID]): ')
    print('\n === BI-LSTM IS THINKING... === \n')
    predict_comment_sentiment(prompt)

if __name__ == '__main__':
    main()