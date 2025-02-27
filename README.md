# AI_IMAGE_CLASSIFIER
# AI-Generated Image Detection

This project is a Flask-based web application that allows users to upload images and detect whether they are AI-generated using the Sightengine API.

## Features
- Upload an image in PNG, JPG, or JPEG format.
- The image is analyzed using the Sightengine API.
- The result indicates whether the image is AI-generated or not.
- Simple and user-friendly UI.

## Technologies Used
- **Flask** - Web framework for Python
- **Sightengine API** - Image analysis
- **HTML, CSS, JavaScript** - Frontend
- **Werkzeug** - Secure file handling

## Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python (3.x)
- Pip (Python package manager)

### Clone the Repository
```bash
 git clone https://github.com/yourusername/ai-image-detection.git
 cd ai-image-detection
Install Dependencies
 pip install -r requirements.txt

Run the Application
 python app.py 
 The application will be available at http://127.0.0.1:5000/.

Project Structure
├── static/
│   ├── uploads/        # Folder for uploaded images
│   ├── css/            # CSS styles (if any)
├── templates/
│   ├── index.html      # Main webpage template
├── app.py              # Main Flask application
├── requirements.txt    # List of dependencies
├── README.md           # Project documentation

API Details
 The project uses Sightengine API to detect AI-generated images.
 API Endpoint: https://api.sightengine.com/1.0/check.json

Required Parameters:
 api_user: Your API user ID
 api_secret: Your API secret key
 models: Set to 'genai' for AI image detection

How It Works
 User uploads an image.
 The image is saved in the static/uploads/ directory.
 The image is sent to the Sightengine API for analysis.
 The API returns a response with an AI-generated score.
 The application determines if the image is AI-generated based on the score (> 0.5 is considered AI-generated).
 The result is displayed on the webpage.
 Example API Response
 {
   "status": "success",
   "type": {
     "ai_generated": 0.8
   }
 }

Future Enhancements
 Improve UI design
 Allow multiple image uploads
 Store analysis history
 Support additional AI detection models

License
 This project is licensed under the MIT License.

Author
 Your Name - Riyanirmalkumar

