from keras.applications import ResNet50 
from keras.layers import Flatten, Dense, Dropout 
from keras.models import Model
from keras.optimizers import SGD
from tensor.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze early layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Add custom classifier
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

# Create full model 
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(loss='categorical_crossentropy', 
              optimizer='rmsprop',
              metrics=['accuracy'])

# Create generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2, 
                                    zoom_range=0.2,
                                    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

# Update the path format for Windows
train_generator = train_datagen.flow_from_directory(
    'C:/Users/yashs/Desktop/data/train',  # Corrected path format
    target_size=(224, 224),
    batch_size=2,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    'C:/Users/yashs/Desktop/data/validation',  # Corrected path format
    target_size=(224, 224),
    batch_size=2,
    class_mode='categorical')

# Train classifier layers
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Changed from hardcoded value
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)  # Changed from hardcoded value
)

# Unfreeze deeper layers
for layer in base_model.layers[-6:]:
    layer.trainable = True

# Recompile and fine-tune model
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),  # Updated to learning_rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Changed from hardcoded value
    epochs=10,
    validation_data=validation_generator,
    validation_steps=len(validation_generator)  # Changed from hardcoded value
)
