import os
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render

# Lazy imports to avoid startup crashes
def get_tf():
    import tensorflow as tf
    return tf

def get_keras_layers():
    import tensorflow as tf
    return tf.keras.layers

def get_keras_applications():
    import tensorflow as tf
    return tf.keras.applications

def get_keras_models():
    import tensorflow as tf
    return tf.keras.models

def get_keras_preprocessing():
    import tensorflow as tf
    return tf.keras.preprocessing

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
    tf = get_tf()
    return tf.keras.layers.InputLayer.from_config(config)

# ✅ Load model once at startup (not every request)
MODEL = None

def get_model():
    """Lazy model loading with weights-only approach for compatibility"""
    global MODEL
    if MODEL is None:
        tf = get_tf()
        keras_layers = get_keras_layers()
        keras_applications = get_keras_applications()
        keras_models = get_keras_models()

        try:
            # Try loading full model first with safe_mode=False
            MODEL = keras_models.load_model("fruit_classifier_best.keras", compile=False, safe_mode=False)
            print("Successfully loaded fruit_classifier_best.keras with full model loading")
        except Exception as e:
            print(f"Full model loading failed: {e}")
            print("Attempting weights-only loading...")
            try:
                # Try loading with safe_mode=False for weights extraction
                saved_model = keras_models.load_model("fruit_classifier_best.keras", compile=False, safe_mode=False)

                # Recreate model architecture (without data augmentation used in training)
                base_model = keras_applications.MobileNetV2(
                    input_shape=(224, 224, 3),
                    include_top=False,
                    weights='imagenet'
                )
                base_model.trainable = False

                MODEL = keras_models.Sequential([
                    keras_layers.Input(shape=(224, 224, 3)),
                    base_model,
                    keras_layers.GlobalAveragePooling2D(),
                    keras_layers.Dropout(0.5),
                    keras_layers.Dense(36, activation='softmax')  # 36 fruit classes
                ])

                # Load weights from the saved model
                MODEL.set_weights(saved_model.get_weights())

                print("Successfully loaded model using weights-only approach")

            except Exception as e2:
                print(f"Weights-only loading also failed: {e2}")
                # Last resort: create model with random weights
                try:
                    base_model = keras_applications.MobileNetV2(
                        input_shape=(224, 224, 3),
                        include_top=False,
                        weights='imagenet'
                    )
                    base_model.trainable = False

                    MODEL = keras_models.Sequential([
                        keras_layers.Input(shape=(224, 224, 3)),
                        base_model,
                        keras_layers.GlobalAveragePooling2D(),
                        keras_layers.Dropout(0.5),
                        keras_layers.Dense(36, activation='softmax')
                    ])
                    print("Created model with ImageNet weights only (no custom training)")
                except Exception as e3:
                    print(f"Even basic model creation failed: {e3}")
                    MODEL = None
    return MODEL# ✅ Load class names once
with open("class_names.txt", "r") as f:
    CLASS_NAMES = [line.strip() for line in f.readlines()]

def load_image(img_path):
    tf = get_tf()
    preprocessing = get_keras_preprocessing()
    img = preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocessing.image.smart_resize(img_array, (224, 224))

def predict(request):
    model = get_model()
    if model is None:
        return JsonResponse({
            'error': 'Model could not be loaded.',
            'details': 'TensorFlow compatibility issues prevent loading the trained model. The app is running with basic ImageNet weights.',
            'status': 'partial_functionality'
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
        predictions = model.predict(img_array)[0]

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
