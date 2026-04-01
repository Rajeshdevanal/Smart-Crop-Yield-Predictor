# Smart Crop Yield Predictor

Smart Crop Yield Predictor is a Flask web application that uses machine learning to estimate crop yield based on field and soil attributes. The app supports multiple languages and helps farmers make more informed decisions by predicting agricultural yield in kg/hectare.

## Features

- Interactive Flask web interface
- Crop yield prediction using pre-trained ML models
- Multilingual support (English, Hindi, Spanish, Portuguese, French, Arabic, Chinese, Bengali)
- Uses fertilizer, temperature, nitrogen, phosphorus, and potassium inputs
- Model selection and result display on the dashboard

## Project Structure

- `app.py` - Flask application and prediction logic
- `templates/` - HTML pages for the web app
- `static/` - CSS and image assets
- `models/` - Pretrained model files and metadata
- `data/` - Dataset and prediction history
- `notebooks/` - Analysis and model training notebooks

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Rajeshdevanal/Smart-Crop-Yield-Predictor.git
   cd Smart-Crop-Yield-Predictor
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install flask numpy joblib
   ```

   If you use a `requirements.txt` file in the future, run:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

```bash
python app.py
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Usage

1. Choose a language from the dropdown
2. Enter fertilizer, temperature, nitrogen, phosphorus, and potassium values
3. Click **Predict Crop Yield**
4. Review the predicted yield and recommendations

## Deployment

This is a Flask application and cannot be hosted directly on GitHub Pages, because GitHub Pages only supports static sites.

To host the app from your GitHub repository, use a Python-friendly hosting service such as:

- Render
- Railway
- Heroku
- Azure App Service

This repo includes a `requirements.txt` file and a `Procfile` so the app can be deployed to platforms that support Python web apps.

If you want to host the website from GitHub, keep the repository on GitHub and connect it to one of those services for automatic deployment.

## Notes

- The application uses pre-trained machine learning models stored in the `models/` directory.
- The prediction is intended for demonstration and should be validated with actual field data before relying on it for agricultural decisions.

## License

This repository is available under the terms of the MIT License.
