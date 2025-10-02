import os
import numpy as np
import tensorflow as tf
from django.http import JsonResponse
from django.shortcuts import render
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ✅ Load model once at startup (not every request)
MODEL = tf.keras.models.load_model("fruit_classifier_finetuned.keras")

# ✅ Load class names once
with open("class_names.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def predict(request):
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
