This project requires a file called `reviews.csv' with all the reviews data to analyse.
It's not commited to safeguard the data.

Current accuracy:
Training Accuracy: 91.68%
Testing Accuracy: 87.97%

Suggested next steps to improve the model:

- generar una nueva feature usando BERT o un LLM (u otra cosa) para generar una columna de positividad o negatividad de la review
  1- feature engineering para generar más features
  2- plot tokens to classes to see if I find any clusters or relationships.
  4- use mutual information to find out what features (tokens) are more useful and what are not so much
  5-review all plot types and check which one might be useful here
  6- apply clustering or PCA to generate new features if possible
- apply PCA
- apply target encoding
- plot the training loss versus validation loss as seen in https://www.kaggle.com/code/ryanholbrook/stochastic-gradient-descent:

PLOT ONLY TRAINING LOSS
history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5. You can change this to get a different view.
history_df.loc[5:, ['loss']].plot();

PLOT EVOLUTION OF TRAINING
# YOUR CODE HERE: Experiment with different values for the learning rate, batch size, and number of examples
learning_rate = 0.05
batch_size = 32
num_examples = 256

animate_sgd(
learning_rate=learning_rate,
batch_size=batch_size,
num_examples=num_examples,
# You can also change these, if you like
steps=50, # total training steps (batches seen)
true_w=3.0, # the slope of the data
true_b=2.0, # the bias of the data
)

- experiment with different architectures for the deep learning model. Also try out Dropout and Batch Normalization layers.
- investigate deeper into the activation layer concept and see how to use it in our problem
- Data augmentation: changing a text that has the same meaning will teach the model to ignore those subleties.
- Using ConvNet layers might help discovering patterns of sentiment in the text.
- Trying hybrid methods: boosted hybrids and stacked hybrids.
