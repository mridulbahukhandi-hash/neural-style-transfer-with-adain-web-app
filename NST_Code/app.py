import os
from pathlib import Path
from uuid import uuid4

import torch
from flask import Flask, render_template, send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from werkzeug.utils import secure_filename
from wtforms import FileField, FloatField, HiddenField, SubmitField
from wtforms.validators import NumberRange

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
EXAMPLES_FOLDER = BASE_DIR / 'examples'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
secret_key = os.environ.get('SECRET_KEY')
if not secret_key and os.environ.get('FLASK_ENV') == 'production':
    raise RuntimeError('SECRET_KEY must be configured in production.')
app.config['SECRET_KEY'] = secret_key or 'local-development-key'
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
Bootstrap(app)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

class UploadForm(FlaskForm):
    content = FileField('Content Image')
    style = FileField('Style Image')
    content_path = HiddenField()
    style_path = HiddenField()
    alpha = FloatField('Alpha', default=1.0, validators=[NumberRange(min=0, max=1)])
    submit = SubmitField('Transfer Style')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = VGGEncoder(str(BASE_DIR / 'vgg_normalised.pth')).to(device)
decoder = Decoder().to(device)
decoder_path = BASE_DIR / 'experiment' / 'final_exp' / 'decoder_final.pth'
decoder.load_state_dict(torch.load(decoder_path, map_location=device))

encoder.eval()
decoder.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def stored_filename(value):
    """Return a safe filename from a previously uploaded form value."""
    if not value or Path(value).name != value or secure_filename(value) != value:
        return None
    return value


def save_upload(upload, previous_filename, label):
    """Save a new upload or safely reuse the previous upload for a field."""
    if not upload or not upload.filename:
        return stored_filename(previous_filename)
    if not allowed_file(upload.filename):
        raise ValueError(f'{label} must be a PNG, JPG, or JPEG image.')

    original_name = secure_filename(upload.filename)
    if not original_name:
        raise ValueError(f'{label} has an invalid filename.')

    filename = f'{uuid4().hex}_{original_name}'
    upload.save(UPLOAD_FOLDER / filename)
    return filename

def style_transfer(content_image, style_image, encoder, decoder, alpha, device):
    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    content_image = content_transform(content_image).unsqueeze(0).to(device)
    style_image = style_transform(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_image, is_test=True)
        style_feats = encoder(style_image, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)

        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_image = decoder(stylized_feats)

    return stylized_image


def save_image(image, path):
    image = image.cpu().clone()
    image = image.squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    result_image = None
    content_filename = None
    style_filename = None
    error = None

    if form.validate_on_submit():
        try:
            content_filename = save_upload(form.content.data, form.content_path.data, 'Content image')
            style_filename = save_upload(form.style.data, form.style_path.data, 'Style image')

            if not content_filename or not style_filename:
                raise ValueError('Please upload both a content image and a style image.')

            form.content_path.data = content_filename
            form.style_path.data = style_filename
            content_image = Image.open(UPLOAD_FOLDER / content_filename).convert('RGB')
            style_image = Image.open(UPLOAD_FOLDER / style_filename).convert('RGB')
            stylized_image = style_transfer(
                content_image,
                style_image,
                encoder,
                decoder,
                form.alpha.data,
                device,
            )

            result_filename = f'stylized_{uuid4().hex}_{content_filename}'
            save_image(stylized_image, UPLOAD_FOLDER / result_filename)
            result_image = result_filename
        except (OSError, RuntimeError, UnidentifiedImageError, ValueError) as exc:
            app.logger.warning('Style transfer failed: %s', exc)
            error = str(exc)
    elif form.errors:
        error = '; '.join(
            message
            for field_errors in form.errors.values()
            for message in field_errors
        )

    return render_template('index.html', form=form, result_image=result_image, content_image=content_filename,
                           style_image=style_filename, error=error)


@app.route('/uploads/<filename>')
def send_image(filename):
    return send_from_directory(str(UPLOAD_FOLDER), filename)


@app.route('/examples/<path:filename>')
def send_example(filename):
    return send_from_directory(str(EXAMPLES_FOLDER), filename)


if __name__ == '__main__':
    app.run(
        host=os.environ.get('HOST', '127.0.0.1'),
        port=int(os.environ.get('PORT', '5000')),
        debug=os.environ.get('FLASK_DEBUG', '').lower() == '1',
    )





