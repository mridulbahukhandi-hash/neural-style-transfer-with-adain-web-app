# Neural Style Transfer with AdaIN

A Flask web app that combines a content image with a style reference using
Adaptive Instance Normalization (AdaIN).

## What this project contains

The neural style-transfer encoder, decoder architecture, and AdaIN operation
are based on the open-source PyTorch implementation
[pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) by naoto0804.
That project is licensed under the MIT License; the attribution and license
notice are included in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The project-specific work in this repository is the Flask application,
upload workflow, alpha control, result handling, web interface, example
gallery, and deployment configuration.

## Contributions and provenance

The web application and interface were assembled and customized for this
project. The neural network architecture and AdaIN operation are credited
above rather than presented as original work.

Before redistributing the repository, verify the licenses or permissions for
the image files in `NST_Code/content_data`, `NST_Code/style_data`,
`NST_Code/examples`, and `Demo_IO_Images`, as well as the downloaded model
weights. Their individual sources are not recorded in the original project.
Do not claim these assets are original until that verification is complete.

## Run locally

```bash
cd NST_Code
pip install -r ../requirements.txt
python app.py
```

Open `http://localhost:5000` in your browser, then upload a content image and
a style image. The style-strength control accepts values from `0` (content
image) to `1` (full AdaIN stylization).

For deployment, set a strong `SECRET_KEY` environment variable. The
application refuses to start with the local fallback when `FLASK_ENV=production`.

The application uses the trained weights in
`NST_Code/experiment/final_exp/decoder_final.pth` and
`NST_Code/vgg_normalised.pth`. These binary model files are provided for
inference; training is available through `NST_Code/train.py`.

## Project structure

- `NST_Code/app.py` - Flask application and inference workflow.
- `NST_Code/utils/` - model definitions and AdaIN utilities derived from the
  credited implementation.
- `NST_Code/train.py` - decoder training entry point.
- `NST_Code/templates/` - the project-specific web interface.
- `Demo_IO_Images/` - example inputs and outputs.

## Limitations

The demo loads the model into memory when the application starts and is
intended for single-image experimentation. Production deployments should use
a suitable upload store, authentication/rate limiting, and a production
secret key.