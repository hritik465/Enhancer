import os
import re
import io
import base64
import numpy as np
from flask import Flask, render_template, request
from PIL import Image, ImageEnhance
import cv2
import fingerprint_enhancer  # Load the fingerprint enhancer library

app = Flask(__name__)

# List to store student data
students_data = []

# Function to sanitize filenames
def sanitize_filename(filename):
    filename = re.sub(r'[\\/*?:"<>|()]+', "", filename)  # Remove invalid characters
    filename = filename.replace(" ", "_")  # Replace spaces with underscores
    return filename

# Function to enhance fingerprint images and return base64 encoded data
def enhance_image_in_memory(image_data):
    # Load the image from memory
    image_stream = io.BytesIO(image_data)
    pil_image = Image.open(image_stream)

    # Convert the image to OpenCV format
    open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray_img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Step 1: Enhance the fingerprint image using the fingerprint_enhancer library
    enhanced_img = fingerprint_enhancer.enhance_fingerprint(gray_img)

    # Step 2: Convert the boolean output to uint8 (since OpenCV expects uint8 for thresholding)
    enhanced_img_uint8 = (enhanced_img * 255).astype('uint8')  # Convert boolean array to uint8 (0 or 255)

    # Step 3: Apply thresholding to convert the background to white and fingerprint to black
    _, out_thresh = cv2.threshold(enhanced_img_uint8, 127, 255, cv2.THRESH_BINARY_INV)

    # Step 4: Convert the thresholded image back to Pillow format for further processing
    pil_enhanced_image = Image.fromarray(out_thresh)

    # Step 5: Enhance the image contrast using Pillow
    enhancer = ImageEnhance.Contrast(pil_enhanced_image)
    enhanced_image = enhancer.enhance(3)

    # Step 6: Flip the image horizontally
    enhanced_image_flipped = enhanced_image.transpose(Image.FLIP_LEFT_RIGHT)

    # Step 7: Create a white background image with the same size as the enhanced image
    width, height = enhanced_image_flipped.size
    white_background = Image.new('RGB', (width, height), (255, 255, 255))  # Create a white background

    # Step 8: Paste the enhanced image onto the white background
    enhanced_image_flipped = enhanced_image_flipped.convert("L")  # Convert to grayscale for transparency handling
    white_background.paste(enhanced_image_flipped, (0, 0))

    # Step 9: Save the final image to a BytesIO object (in-memory)
    enhanced_img_io = io.BytesIO()
    white_background.save(enhanced_img_io, format='PNG')  # Save as PNG format in memory
    enhanced_img_io.seek(0)

    # Encode the image to base64 to display it inline on the webpage
    encoded_img = base64.b64encode(enhanced_img_io.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{encoded_img}"

# Route for homepage
@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form.get('userName')
        user_id = request.form.get('userId')

        enhanced_images = []
        for i in range(1, 4):  # 3 fingerprint images
            file = request.files.get(f'fingerprintFile{i}')
            if file:
                image_data = file.read()
                encoded_image = enhance_image_in_memory(image_data)
                enhanced_images.append(encoded_image)

        # Add the current student's data to the list
        students_data.append({'name': name, 'user_id': user_id, 'enhanced_images': enhanced_images})

        # Pass the entire list of students' data to display on the webpage
        return render_template('index.html', students=students_data)

    return render_template("index.html", students=[])

if __name__ == "__main__":
    app.run(debug=True)
