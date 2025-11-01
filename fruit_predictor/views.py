import os
import numpy as np
import tensorflow as tf
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Custom function to handle InputLayer deserialization
def custom_input_layer_deserializer(config):
    # Remove batch_shape if it exists and replace with shape
    if 'batch_shape' in config:
        config = config.copy()
        batch_shape = config.pop('batch_shape')
        if batch_shape[0] is None:  # Remove batch dimension
            config['shape'] = batch_shape[1:]
        else:
            config['shape'] = batch_shape
    return tf.keras.layers.InputLayer.from_config(config)

# ✅ Load model once at startup (not every request)
MODEL = None
print("Model loading disabled due to compatibility issues with current TensorFlow version")
print("The saved models were created with an older TensorFlow/Keras version")
print("To fix this, you would need to:")
print("1. Retrain the model with TensorFlow 2.13.0, or")
print("2. Use a TensorFlow version compatible with the saved models, or")
print("3. Convert the models to a compatible format")

# ✅ Load class names once
with open("class_names.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict(request):
    if MODEL is None:
        return JsonResponse({
            'error': 'Model not loaded due to compatibility issues.',
            'details': 'The saved models were created with an older TensorFlow/Keras version that is incompatible with the current version (2.13.0). The models contain InputLayer configurations with batch_shape parameters that are not supported in the current version.',
            'solutions': [
                '1. Retrain the model using TensorFlow 2.13.0',
                '2. Downgrade TensorFlow to a compatible version',
                '3. Convert the model to SavedModel format',
                '4. Use model weights only and recreate the architecture'
            ]
        })
        
    if request.method == "POST" and 'image' in request.FILES:
        img = request.FILES['image']
        temp_dir = os.path.join(os.getcwd(), 'temp')
        os.makedirs(temp_dir, exist_ok=True)  # ✅ ensure temp dir exists

        img_path = os.path.join(temp_dir, img.name)
        with open(img_path, 'wb') as f:
            for chunk in img.chunks():
                f.write(chunk)

        # Process image
        img_array = load_image(img_path)
        predictions = MODEL.predict(img_array)[0]

        # Top-3 predictions
        top3_indices = predictions.argsort()[-3:][::-1]
        results = []
        for i in top3_indices:
            class_name = CLASS_NAMES[i]
            probability = predictions[i]
            results.append({
                "class": class_name,
                "probability": "{:.1f}%".format(float(probability) * 100)
            })

        # ✅ Optionally remove temp file
        try:
            os.remove(img_path)
        except:
            pass

        return JsonResponse({'results': results})

    return render(request, "predict.html")
