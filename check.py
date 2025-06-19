from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model(r"D:\fresh_rot_fruits_model\model1.keras")

# Load and preprocess the image
img = image.load_img(r"D:\ds_files\Fresh_rot_fruits.pic\dataset\freshapples\rotated_by_15_Screen Shot 2018-06-08 at 5.21.31 PM.png", target_size=(224, 224))
def predict(model, img):
    # Preprocess the image
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img_array)

    # Get predicted class and confidence
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)

    # Display the image using matplotlib

    return predicted_class, confidence

class_names = ['freshapples', 'freshbanana', 'rottenapples', 'rottenbanana']
p,c = predict(model,img)
print(p,c)
