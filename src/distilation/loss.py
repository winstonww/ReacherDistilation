import tensorflow as tf

def kl_loss(
    s_pdflat_batch,
    t_pdflat_batch,
    pdtype):

    s = pdtype.pdfromflat(s_pdflat_batch)
    t = pdtype.pdfromflat(t_pdflat_batch)

    return tf.reduce_sum(t.logstd - s.logstd + 
            (tf.square(s.std) + tf.square(s.mean - t.mean)) / 
            (2.0 * tf.square(t.std)) - 0.5, axis=[2,1,0], name="kl_loss" )
