import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(64, 64, 1)):
    """
    Construieste arhitectura CNN.
    """
    model = models.Sequential([
        # Partea 1: Extragere trasaturi
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Partea 2: Clasificare
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid') 
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()