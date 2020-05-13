from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from tensorflow import keras
import os
import re
import time
logger = tf.get_logger()
logger.propagate = False
tf.logging.set_verbosity(tf.logging.INFO)
train_df = pd.read_json('train1.json')
# train_df = pd.read_json('test.jsonl',lines=True)


train, test = train_test_split(train_df, test_size=0.05, random_state=42)
DATA_COLUMN = 'text'
LABEL_COLUMN = 'stars'
label_list = [1,2,3,4,5]

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
    return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
    """Creates a classification model."""

    bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
    bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
    bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    # Create our own layer to tune for politeness data.
    output_weights = tf.get_variable(
      "output_weights", [num_labels-1, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
      "output_bias", [num_labels-1], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probs = tf.nn.sigmoid(logits)
        log_probs = tf.math.log(tf.clip_by_value(probs, 1e-10, 1.0))
        minus_log_probs = tf.math.log(tf.clip_by_value(1-probs, 1e-10, 1.0))


        embed = tf.constant([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0],[1,1,1,1]],tf.float32)
        embed_labels = tf.nn.embedding_lookup(embed,labels)
        
#         constraints_loss = constraints_loss1 + constraints_loss2 + constraints_loss3 + constraints_loss4
        cond = tf.cast(probs < 0.5, tf.int32)
        cond = tf.concat([cond,tf.ones([tf.shape(cond)[0],1], tf.int32)],axis = -1)
        predicted_labels = tf.argmax(cond, axis=-1, output_type=tf.int32)
#         predicted_labels = tf.where(tf.equal(predicted_labels, 0), tf.ones_like(predicted_labels), predicted_labels)
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = tf.reduce_sum(embed_labels*log_probs + (1-embed_labels)*minus_log_probs, axis=-1)
        loss = -tf.reduce_mean(per_example_loss)
        return (loss, predicted_labels, probs,predicted_labels)
# model_fn_builder actually creates our model function
# using the passed parameters for num_labels, learning_rate, etc.
def model_fn_builder(num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)
    
        # TRAIN and EVAL
        if not is_predicting:
            (loss, predicted_labels, probs,log) = create_model(
                  is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics. 
            def metric_fn(label_ids, predicted_labels):
                accuracy = tf.compat.v1.metrics.accuracy(label_ids, predicted_labels)
                return {
                      "accuracy": accuracy,
                }
            eval_metrics = metric_fn(label_ids, predicted_labels)
            accuracy = tf.compat.v1.metrics.accuracy(label_ids, predicted_labels)
            tf.summary.scalar('accuracy', accuracy[1])
            if mode == tf.estimator.ModeKeys.TRAIN:
                probs
                training_hooks=[
                    tf.estimator.LoggingTensorHook(
                    tensors={'accuracy':accuracy[1],"probs":probs,"log":log,"labels":label_ids
                            }, every_n_iter=SAVE_SUMMARY_STEPS)
                ]
                return tf.estimator.EstimatorSpec(mode=mode,
                  loss=loss,
                  train_op=train_op,
                  eval_metric_ops=eval_metrics,
                  training_hooks=training_hooks)
            else:
                return tf.estimator.EstimatorSpec(mode=mode,
                  loss=loss,
                  eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, p) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
            predictions = {
              'probabilities': p,
              'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn
# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128

# Use the InputExample class from BERT's run_classifier code to create examples from the data
train_InputExamples = train_df.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)



# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
# Compute train and warmup steps from batch size
# These hyperparameters are copied from this colab notebook (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 2.0
# Warmup is a period of time where hte learning rate 
# is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100
# Compute # train and warmup steps from batch size
# train1_step = 30013
train1_step = 0


num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS) +train1_step
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION) 
# Specify outpit directory and number of checkpoint steps to save
if not(os.path.exists('bert')):
        os.makedirs('bert')
model_dir = "crossentropy2_sigmoid"
# model_dir = "test"

model_dir = os.path.join('bert', model_dir)
if not(os.path.exists(model_dir)):
    os.makedirs(model_dir)
run_config = tf.estimator.RunConfig(
    model_dir=model_dir,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
    log_step_count_steps = SAVE_SUMMARY_STEPS)
model_fn = model_fn_builder(
  num_labels=len(label_list),
  learning_rate=LEARNING_RATE,
  num_train_steps=num_train_steps,
  num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(
  model_fn=model_fn,
  config=run_config,
  params={"batch_size": BATCH_SIZE})




# Create an input function for training. drop_remainder = True for using TPUs.
train_input_fn = bert.run_classifier.input_fn_builder(
    features=train_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False)
test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False)
print(f'Beginning Training!')

current_time = datetime.now()
eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn,throttle_secs=60,steps=None)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

print("Training took time ", datetime.now() - current_time)