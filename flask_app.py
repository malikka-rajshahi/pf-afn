from flask import Flask, render_template, request, flash, url_for, session, send_from_directory
from PIL import Image
import os
import traceback
import numpy as np
from main import Tryon
from data.cloth_edge import GenerateEdge
app = Flask(__name__)

DATAROOT = "dataset"
CLOTHING_PATH = os.path.join(DATAROOT, "test_clothes")
CLOTHING_INP_PATH = os.path.join(DATAROOT, "input_cloth")
IMG_PATH = os.path.join(DATAROOT, "test_img")
EDGE_PATH = "dataset/test_edge/input_cloth.jpg"
# PAIRS_PATH = "SD-VITON/dataroot/test_pairs.txt"
OUTPUT_PATH = DATAROOT

app.config['MAX_LENGTH'] = 200 * 1024 * 1024  # 200MB limit
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.jpeg']
app.secret_key = 'SD-VITON-mitcarbon-2664825'

# def test_pairs(image_name, clothing):
#     f = open(PAIRS_PATH, 'w')
#     f.write(f'{image_name} {clothing}')
#     f.close()

@app.route('/dataset/<path:filename>')
def get_dataroot_image(filename):
    return send_from_directory(DATAROOT, filename)

@app.route('/dataset/<path:filename>')
def get_output_image(filename):
    return send_from_directory(OUTPUT_PATH, filename)

@app.route("/", methods=['GET', 'POST'])
def main():
    image_name = None
    output = None
    if 'uploaded' not in session: session['uploaded'] = False

    if request.method == 'POST':
        if 'file' in request.files:
            input_file = request.files.get('file')
            if input_file and input_file.filename != '':
                filename = input_file.filename
                file_ext = os.path.splitext(filename)[1].lower()

                if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                    flash('Invalid file format. Please upload JPG, PNG, or JPEG.')
                else:
                    try:
                        image = Image.open(input_file.stream)
                        # image = image.resize((768, 1024))
                        # print(f'Image size: {image.size}')

                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        print(image.mode)
                        image.save(os.path.join(IMG_PATH, 'input_image.jpg'))

                        session['uploaded_image'] = 'input_image.jpg'
                        image_name = 'input_image.jpg'
                        flash('Image uploaded successfully!')
                        session['uploaded'] = True

                        clothing_images = os.listdir(CLOTHING_PATH)
                        print(clothing_images)
                        return render_template("flask_display.html", clothing_images=clothing_images, image=image_name)
                    except IOError:
                        flash('Invalid image file.')

        if 'generate' in request.form and session['uploaded']:
            try:
                image_name = session.get('uploaded_image')
                selected_cloth = request.form.get('selected_cloth')
                print("Image, Cloth : ", image_name, selected_cloth)

                try:
                    tryon = Tryon()
                    inp_image = Image.open(f'{IMG_PATH}/input_image.jpg')
                    cloth_image = Image.open(f'{CLOTHING_PATH}/{selected_cloth}')
                    cloth_image.save(f'{CLOTHING_INP_PATH}/input_cloth.jpg')
                    edge_gen = GenerateEdge(f'{CLOTHING_INP_PATH}/input_cloth.jpg')
                    edge_gen.process_images(EDGE_PATH)
                    edge_image = Image.open(EDGE_PATH)
                    edge_image.save(EDGE_PATH)
                    tryon.send_here(inp_image, cloth_image, edge_image)
                    output = "out.jpg"
                    os.remove('/dataset/test_clothes/input_cloth.jpg')

                    # sizing output image
                    out_img = Image.open('dataset/out.jpg')
                    in_img = Image.open('dataset/test_img/input_image.jpg')
                    width, height = in_img.size
                    out_img = out_img.resize((width, height))
                    out_img.save('dataset/out.jpg')
                    print(out_img.size)

                except Exception as e:
                    print(f"Error: {e}")
            except Exception as e:
                flash(f'An error occurred: {str(e)}')
                traceback.print_exc()
    
    clothing_images = os.listdir(CLOTHING_PATH)
    return render_template("flask_display.html", clothing_images=clothing_images, image=image_name, output=output)

if __name__ == "__main__": app.run(host='0.0.0.0', port='5000')
