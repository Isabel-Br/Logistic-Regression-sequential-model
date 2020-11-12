import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0: Print all tensorflow debug messages. 1: Filter out INFO messages.
                                          # 2: Filter out INFO and WARNING messages. 3: Filter out all messages.
import tensorflow as tf
from Preprocess_data import *
# from MC_preprocessing import save_dataframe
# save_dataframe()

"""
Steps:
Import and preprocess data.
Build neural network.
Train neural network.
Export and postprocess results.
"""

# SEQUENTIAL MODEL
# FUNCTIONAL API
# TENSORFLOW v.1

# IMPORT AND PREPROCESS DATA #

df = read_data()
x, y = sort_data(df)
n_validation = int(0.15 * len(x))  # Decide size of validation set
n_test = 32  # Decide size of test set
x_train, y_train, x_val, y_val, x_test, y_test = create_sets(x, y, n_validation, n_test)  # Must be arrays, not lists!

plot_data(x, y)  # View data


# BUILD NEURAL NETWORK #
learning_rate = 0.01
dropout_rate = 0.1

learning_rate_decay = tf.keras.optimizers.schedules.ExponentialDecay(
    learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, input_shape=(3,), activation='relu'),  # Input shape: Parameter dimensions.
            # For dim(input) = (num_examples, num_variables), input_shape = num_variables. Ignore batch size.
            # If num_variables only has 1 dimension, use input_shape = (num_variables, ) or input_shape = num_variables
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(rate=dropout_rate),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(rate=dropout_rate),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Last layer uses sigmoid for binary classification
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate_decay),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])  # Can use more metrics, like tf.keras.metrics.Precision()

print(model.summary())  # Print model structure


# TRAIN NEURAL NETWORK #
print('Start training... \n', 'Learning rate = ', learning_rate, 'Dropout = ', dropout_rate)
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_val, y_val))
print('...training finished.\n\n')

print('Test data evaluation:')
model.evaluate(x_test, y_test)

# EXPORT AND POSTPROCESS RESULTS #
model.save('saved_model.h5')

model = tf.keras.models.load_model('saved_model.h5')

print('Loaded model evaluation:')
loss, acc = model.evaluate(x_test, y_test)
print("Model accuracy: {:5.2f}%".format(100 * acc))

print("Generate predictions and plot results")
predictions = model.predict(x_test)

# Plot results
plot_intervals = [var_list[0] for var_list in x_test]
plot_time_viewing_answer = [var_list[2] for var_list in x_test]

fig, axes = plt.subplots(nrows=1, ncols=2)
fig.tight_layout(h_pad=0, w_pad=3)

axes[0].scatter(x=plot_intervals, y=plot_time_viewing_answer, c=y_test, cmap='RdYlGn')
axes[1].scatter(x=plot_intervals, y=plot_time_viewing_answer, c=predictions, cmap='RdYlGn')

# Title and labels
fig.suptitle('True labels and predictions comparison')  # Global title
plt.subplots_adjust(top=0.85, bottom=0.15, right=0.92, left=0.13)

axes[0].set_title('Test data')
axes[0].set_xlabel('Interval (hours)')
axes[0].set_ylabel('Time viewing answer (seconds)')

axes[1].set_title('Predictions')
axes[1].set_xlabel('Interval (hours)')
axes[1].set_ylabel('Time viewing answer (seconds)')

plt.show()
