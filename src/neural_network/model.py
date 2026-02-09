import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(64, 64, 1)):
    
    model = models.Sequential([
        # 1. NORMALIZARE AUTOMATA
        # Modelul va imparti singur pixelii la 255.
        # Asta rezolva problema cu serverul care dadea rezultate gresite.
        layers.Rescaling(1./255, input_shape=input_shape),

        # 2. AUGMENTARE DATE (Doar la antrenare)
        # Previne overfitting-ul (acuratetea falsa de 1.00)
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # 3. EXTRAGERE TRASATURI (CNN)
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu'), 
        layers.MaxPooling2D((2, 2)),
        
        # 4. CLASIFICARE
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Ignora 50% din neuroni random (regularizare)
        layers.Dense(1, activation='sigmoid') # 0 = Defect, 1 = OK (sau invers)
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = build_cnn_model()
    model.summary()