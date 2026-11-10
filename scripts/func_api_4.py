import pickle
import numpy as np
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.backend as K

def aerodynamic_weighted_mse(y_true, y_pred):
    # Separate the true and predicted values
    true_cl, true_cd, true_cm = y_true[:, 0], y_true[:, 1], y_true[:, 2]
    pred_cl, pred_cd, pred_cm = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
    
    # Calculate independent squared errors
    error_cl = K.square(true_cl - pred_cl)
    error_cd = K.square(true_cd - pred_cd)
    error_cm = K.square(true_cm - pred_cm)
    
    # Multiply the tiny Cd and Cm errors so the network cannot ignore them
    weighted_loss = error_cl + (error_cd * 10.0) + (error_cm * 2.0)
    
    return K.mean(weighted_loss)

# ==========================================
# 1. INGESTION DIRECTLY FROM DICTIONARY
# ==========================================
def load_tensors_from_dictionary(filepath='data/clean_airfoil_tensor_dict.pkl'):
    print(f"Loading dictionary from {filepath}...")
    with open(filepath, 'rb') as file:
        airfoil_dict = pickle.load(file)
        
    X_list = []
    y_list = []
    
    for i, (key, item) in enumerate(airfoil_dict.items()):
        input_vector = np.concatenate((item['conditions'], item['cst']))
        output_vector = item['aerodynamics']
        # [TEST CODELINE]: Print the exact concatenated vector for the first entry
        if i == 0:
            print(f"\n--- Tensor Construction Verification for '{key}' ---")
            print(f"Index 0 (Reynolds)    : {input_vector[0]}")
            print(f"Index 1 (AoA)         : {input_vector[1]}")
            print(f"Indices 2-20 (CST)    : {input_vector[2:]}")
            print(f"Final Input Vector    : {input_vector}")
            print(f"Vector Shape          : {input_vector.shape}")
            print(f"Target Outputs (Cl, Cd, Cm): {output_vector}\n")
            
        X_list.append(input_vector)
        y_list.append(output_vector)
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    print(f"Extraction complete. Total samples: {X.shape[0]}")
    return X, y

# ==========================================
# 2. DATA SPLIT (90% Train, 10% Test)
# ==========================================
X, y = load_tensors_from_dictionary()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# ==========================================
# 3. HIGH-CAPACITY FUNCTIONAL ARCHITECTURE
# ==========================================
inputs = Input(shape=(21,), name="Input_FlightConditions_and_CST")

# 1. Expanded Shared Core with He Initialization
x = Dense(64, activation='mish', kernel_initializer='he_normal', name="Core_Dense_1")(inputs)
x = Dense(128, activation='mish', kernel_initializer='he_normal', name="Core_Dense_2")(x)
x = Dense(256, activation='mish', kernel_initializer='he_normal', name="Core_Dense_3")(x)
x = Dense(256, activation='mish', kernel_initializer='he_normal', name="Core_Dense_4")(x)
x = Dense(128, activation='mish', kernel_initializer='he_normal', name="Core_Dense_5")(x)

# 2. Residual Skip Connection (Fusing 128 abstract features + 21 raw features)
merged = Concatenate(name="Skip_Connection_Raw_Geometry")([x, inputs])

# Widened Expert Branches (256 -> 128 -> 64)
# -- Lift Branch --
lift = Dense(256, activation='mish', kernel_initializer='he_normal', name="Lift_Dense_1")(merged)
lift = Dense(128, activation='mish', kernel_initializer='he_normal', name="Lift_Dense_2")(lift)
lift = Dense(64, activation='mish', kernel_initializer='he_normal', name="Lift_Dense_3")(lift)
out_cl = Dense(1, activation=None, name="Target_Cl")(lift)

# -- Drag Branch --
drag = Dense(256, activation='mish', kernel_initializer='he_normal', name="Drag_Dense_1")(merged)
drag = Dense(128, activation='mish', kernel_initializer='he_normal', name="Drag_Dense_2")(drag)
drag = Dense(64, activation='mish', kernel_initializer='he_normal', name="Drag_Dense_3")(drag)
out_cd = Dense(1, activation=None, name="Target_Cd")(drag)

# -- Moment Branch --
moment = Dense(256, activation='mish', kernel_initializer='he_normal', name="Moment_Dense_1")(merged)
moment = Dense(128, activation='mish', kernel_initializer='he_normal', name="Moment_Dense_2")(moment)
moment = Dense(64, activation='mish', kernel_initializer='he_normal', name="Moment_Dense_3")(moment)
out_cm = Dense(1, activation=None, name="Target_Cm")(moment)

outputs = Concatenate(name="Output_AeroCoefficients")([out_cl, out_cd, out_cm])
model = Model(inputs=inputs, outputs=outputs, name="Heavy_Surrogate_PINN")

# ==========================================
# 4. COMPILATION & TIGHTENED CALLBACKS
# ==========================================
# Swapped to Nadam for Nesterov look-ahead momentum
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss=aerodynamic_weighted_mse,  # Injecting the custom multiplier loss
    metrics=['mae']
)

model.summary()

# Slice the learning rate faster and smaller
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.4, 
    patience=6, 
    min_lr=1e-7,
    min_delta=5e-6,
    verbose=1
)

# Extended patience to 30: Let the computer run as long as it needs to hit the target
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=30, 
    min_delta=1e-6, # Demands microscopic improvements to keep training
    restore_best_weights=True,
    verbose=1
)

# ==========================================
# 5. EXECUTION LOOP
# ==========================================
print("\nInitiating high-capacity training loop...")

# Define the batch size as a variable
current_batch_size = 64

start_time = time.time()
history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=1200,          
    batch_size=current_batch_size,   # Inject the variable here     
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

end_time = time.time()
total_seconds = end_time - start_time
minutes = int(total_seconds // 60)
seconds = int(total_seconds % 60)

print(f"\n--- Training Phase Completed ---")
print(f"Total Training Duration: {minutes} minutes and {seconds} seconds.")

# ==========================================
# 6. BLIND TEST EVALUATION & SERIALIZATION
# ==========================================
print("\n--- Evaluating On Unseen Test Set (10% Holdout) ---")
test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)

print(f"Final Test Loss (MSE): {test_mse:.6e}")
print(f"Final Mean Absolute Error (MAE): {test_mae:.6e}")
print(f"Batch Size Used: {current_batch_size}") # New print statement

model_save_path = 'data/clean_airfoil_surrogate_model.keras'
model.save(model_save_path)
print(f"Trained neural network model saved to: {model_save_path}")