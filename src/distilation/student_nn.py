from config import *
import tensorflow as tf

'''
This method constructs the lstm graph.
@params  
        training_input_batch: 
            - Tensor of size [ STEPS_UNROLLED, LSTM_BATCH_SIZE, OBSPACE_SHAPE+PDFLAT_SHAPE ]
            - The input to be fed into the LSTM network. Appending the previous action flat
          param to the observation as input
        initial_state_batch_ph:
            - Tensor of size [ 2, LSTM_BATCH_SIZE, NUM_UNITS ]
            - The initial hidden state of LSTM
@return
        output_batch:
            - Tensor of size [ STEPS_UNROLLED, LSTM_BATCH_SIZE, PDFLAT_SHAPE ]
            - the action to be taken wrt the the input given
        tf.identity(state_batch):
            - The final hidden state of LSTM
'''
def student_lstm_graph( ob_batch_ph, keep_prob_ph, prev_pdflat_batch_ph, initial_state_batch_ph ):

    cell = tf.contrib.rnn.LSTMCell( num_units=NUM_UNITS, name="unique_lstm_cell" )

    ob_dropout_batch = tf.nn.dropout( ob_batch_ph, keep_prob_ph )

    hid_prev_pdflat_batch = tf.layers.dense( prev_pdflat_batch_ph, 32 )

    # combine the dropped out observation, previous reward and previous action to create a flat representing the input
    input_batch = tf.concat( [ ob_dropout_batch, hid_prev_pdflat_batch ], -1 )

    # Initailize state with zero of batch size 1 and type float32
    c_state_batch, m_state_batch = initial_state_batch_ph[0,:,:], initial_state_batch_ph[1,:,:]

    state_batch = tf.tuple( [ c_state_batch, m_state_batch ], name="cm_state" )

    output_batch = []

    for i in range(STEPS_UNROLLED):

        dense, state_batch = cell( input_batch[ i, :, : ], state_batch )
        dense = tf.nn.tanh( tf.layers.dense(dense, 64 ) )
        dense = tf.nn.tanh( tf.layers.dense(dense, 128 ) )
        dense = tf.nn.tanh( tf.layers.dense(dense, 64 ) )
        dense = tf.nn.tanh( tf.layers.dense(dense, 32 ) )
        dense = tf.layers.dense(dense, PDFLAT_SHAPE )
        output_batch.append(dense)

    return tf.stack(output_batch), tf.identity(state_batch)

def student_mlp_graph( training_input_batch ):
    dense = tf.nn.tanh( tf.layers.dense(training_input_batch, 24 ) )
    dense = tf.nn.tanh( tf.layers.dense(dense, 128 ) )
    dense = tf.layers.dense(dense, 128 )
    dense = tf.nn.tanh( tf.layers.dense(dense, 32 ) )
    dense = tf.layers.dense(dense, PDFLAT_SHAPE )
    return dense
