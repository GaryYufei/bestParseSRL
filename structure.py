import numpy as np

def viterbi_decode(score, transition_params, trans_mask, unary_mask):
  trellis = np.zeros_like(score)
  backpointers = np.zeros_like(score, dtype=np.int32)
  trellis[0] = score[0]

  for t in range(1, score.shape[0]):
    mask = (trans_mask[t] == 1).astype(float)
    masked_trans_params = transition_params * mask + (1 - mask) * trans_mask[t]
    v = np.expand_dims(trellis[t - 1], 1) + masked_trans_params
    mask = (unary_mask[t] == 1).astype(float)
    unary_score = score[t] * mask + (1 - mask) * unary_mask[t]
    trellis[t] = unary_score + np.max(v, 0)
    backpointers[t] = np.argmax(v, 0)

  viterbi = [np.argmax(trellis[-1])]
  for bp in reversed(backpointers[1:]):
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()

  viterbi_score = np.max(trellis[-1])
  return viterbi, viterbi_score

def viterbi_decode_nbest(score, transition_params, nbest=3):
  back_points = list()
  nbest_scores_t_history = list()

  (seq_len, tag_size) = score.shape
  score = score.reshape(seq_len, 1, tag_size)
  trans_scores = score[1:, :, :] + transition_params.reshape([1, tag_size, tag_size])

  nbest_scores_t = score[0]
  nbest_scores_t_history.append(np.tile(nbest_scores_t.reshape([tag_size, 1]), [1, nbest]))

  for t, scores_t in enumerate(trans_scores):
    if t == 0:
      scores_t = scores_t + nbest_scores_t.reshape([tag_size, 1])
    else:
      scores_t = np.tile(scores_t.reshape([tag_size, 1, tag_size]), [1, nbest, 1]) \
                         + np.tile(nbest_scores_t.reshape([tag_size, nbest, 1]), [1, 1, tag_size])
      scores_t = scores_t.reshape([tag_size * nbest, tag_size])
    cur_bp = np.argsort(scores_t, axis=0)[-nbest:][::-1, :]
    nbest_scores_t = scores_t[cur_bp, np.tile(np.arange(0, tag_size).reshape([1, tag_size]), [nbest, 1])]
    if t == 0: cur_bp = cur_bp * nbest
    nbest_scores_t = nbest_scores_t.transpose([1, 0])
    cur_bp = cur_bp.transpose([1, 0])
    nbest_scores_t_history.append(nbest_scores_t)
    back_points.append(cur_bp)
  
  seq_score = nbest_scores_t_history[-1].reshape((tag_size * nbest))
  viterbi_score = np.sort(seq_score)[-nbest:][::-1]

  viterbi= [np.argsort(seq_score, axis=0)[-nbest:][::-1]]
  for bp in reversed(back_points):
    bp = bp.reshape((tag_size * nbest))
    viterbi.append(bp[viterbi[-1]])
  viterbi.reverse()
  viterbi = np.concatenate([(np.expand_dims(v, 0) / nbest).astype(np.int32) for v in viterbi], axis=0)
  viterbi = viterbi.transpose([1, 0])
  viterbi = viterbi.tolist()

  return viterbi, viterbi_score
