from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import numpy as np
import tensorflow as tf

categories = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

class ASLPredictionApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')

        # Image display for camera feed
        self.camera_feed = Image(size_hint=(1, 0.8))
        layout.add_widget(self.camera_feed)

        # Button to trigger prediction
        self.predict_button = Button(text='Click to Capture Image', on_press=self.predict_sign)
        layout.add_widget(self.predict_button)

        # Label to display prediction result
        self.prediction_label = Label(text='Letter: ', size_hint=(1, 0.1))
        layout.add_widget(self.prediction_label)

        #load model
        self.model = tf.keras.models.load_model('C:\\Users\\Owner\\app\\model2.h5')  

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_camera_feed, 1.0 / 30.0)

        return layout

    def update_camera_feed(self, *args):
        ret, frame = self.capture.read()

        
        resized_frame = cv2.resize(frame, (400, 400))  

        
        flipped_frame = cv2.flip(resized_frame, 0)

        # Convert BGR frame to RGB for display in Kivy
        converted_frame = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

        # Update the camera feed image
        buf = converted_frame.tostring()
        texture = Texture.create(size=(flipped_frame.shape[1], flipped_frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.camera_feed.texture = texture

    def predict_sign(self, instance):
        # Perform prediction on the captured frame
        ret, frame = self.capture.read()

        # Preprocess the frame for prediction 
        processed_frame = cv2.resize(frame, (28, 28))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        processed_frame = processed_frame / 255.0  # Normalize

        # Reshape and prepare the frame for prediction
        processed_frame = processed_frame.reshape((1, 28, 28, 1))

        # Make predictions using the loaded model
        predictions = self.model.predict(processed_frame)
        predicted_label_index = np.argmax(predictions, axis=1)[0]

        # Display the predicted label in the label widget
        predicted_label = categories[predicted_label_index]
        self.prediction_label.text = f"Predicted Label: {predicted_label}"


if __name__ == '__main__':
    ASLPredictionApp().run()
