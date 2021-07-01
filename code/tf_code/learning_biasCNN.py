#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define a custom train step function.
By MMH.
"""

from tensorflow.contrib.slim.python.slim import learning
#from tensorflow.contrib.slim.python.slim import evaluation
from tensorflow.python.platform import tf_logging as logging


def train_step_fn(sess, train_op, global_step, train_step_kwargs):
  """Function that takes a gradient step and specifies whether to stop.
      (Wrapper function for train_step in learning.py)
  Args:
    sess: The current session.
    train_op: An `Operation` that evaluates the gradients and returns the
      total loss.
    global_step: A `Tensor` representing the global training step.
    train_step_kwargs: A dictionary of keyword arguments (using this here to 
    pass in my eval_op for validation set evaluation, so we can do evaluation
    during training).

  Returns:
    The total loss and a boolean indicating whether or not to stop training.

  """

  # first, run the normal training step (happens every single step)
  total_loss, should_stop = learning.train_step(sess, train_op, global_step, train_step_kwargs)

  # get global step as a value i can log 
  global_step = sess.run(global_step)
 
  # decide whether to reset the streaming evaluation metrics
  should_reset = sess.run(train_step_kwargs['should_reset_eval_metrics'])
  if should_reset:
      # reset counter and total
      logging.info('RESETTING STREAMING EVAL METRICS AT STEP %d\n'% (global_step))
      sess.run([train_step_kwargs['reset_op']])
  
  # decide whether to run evaluation
  should_val = sess.run(train_step_kwargs['should_val'])    
  if should_val:   
      # validate
      logging.info('EVALUATING MODEL AT STEP %d\n'% (global_step))
      sess.run([train_step_kwargs['eval_op']] )

  return [total_loss, should_stop]
  