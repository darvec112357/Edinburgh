a1aa=['DET', 'NOUN', 'ADJ', 'VERB', 'ADP', '.', 'ADV', 'CONJ', 'PRT', 'PRON', 'NUM', 'X']
a1b=2649
a1c=12.061525956381585
a1d='function'
a2a=13
a2b=2.4630366660416674
a3c=2.883894963080882
a3d='DET'
a4a3=0.8941143860297515
a4b1=[('``', '.'), ('My', 'DET'), ('taste', 'NOUN'), ('is', 'VERB'), ('gaudy', 'ADV'), ('.', '.')]
a4b2=[('``', '.'), ('My', 'DET'), ('taste', 'NOUN'), ('is', 'VERB'), ('gaudy', 'ADJ'), ('.', '.')]
a4b3="In the above sentence, the model incorrectly predicts the tag of 'gaudy' as ADV instead of ADJ.This is because 'gaudy' has\nlittle appearance in the corpus and P(ADV|VERB) is more likely than P(ADJ|VERB). Hence ADV is favoured."
a4c=60.73342283709656
a4d=70.21922688308214
a4e=['DET', 'NOUN', 'ADP', 'DET', 'NOUN', 'VERB', 'ADV']
a5t0=0.558440358495796
a5tk=0.6090732698882011
a5b="The word 'he' does not appear in the training set so the model knows nothing about it.\nP(NUM|VERB) is more likely than P(PRON|VERB) in the training set. Hence, the model would favour 'NUM'.\nOccurences of 'he' in unlabeled data will help the model tag it more accurately.T_0 correctly tags 'them' as PRON while T_k\nmislabels it as 'NOUN', because there are much more sentences ending with NOUN than PRON. As training goes deeper, such difference\nin likelyhood will affect predictions."
a6='Use the emission model of the pretrained POS tagger for preterminal rules of PCFG, which includes the\nsmoothed likelihood of all unknown words. Then we can use the PCFG to produce a parse for the given sentence. However, such method may not\nwork as well on sentences that do not contain unknown words, especially when there is significant difference between likelihoods of known words.'
a7="The Brown Corpus(BC) tagset has more tags than the universal tagset. The data will be sparser for training\nwhich will lead to inaccurate predictions using viterbi. For example, for any verbs, universal tagset just needs to\ntag it as 'VERB' while the BC tagset has to classify it into 'VB','VBD' etc. In Q5 where we need to simulate a semi-supervised learning\nwith much smaller training data,this problem will be worsened."
a4full_vit=[[26.782208935674642, 27.175612695180405, 26.83747232781355, 25.69387747486122, 26.069930312266184, 25.33122669020975, 26.602772790690192, 26.7617409338053, 6.7157647918435055, 26.154747513228394, 28.346102791874564, 27.06281489027102], [34.01526253042612, 26.139704758753695, 34.54383920395947, 32.31894404945449, 34.25048137679512, 36.22833529220789, 27.617716596488677, 37.174460980126284, 34.96569792142533, 33.2167214529358, 30.719087535156927, 43.877716268423896], [52.96761801645391, 52.61690686578734, 53.24942252459456, 54.72356597513747, 52.53968502182914, 34.012187579823376, 51.358458806319945, 52.70225028680407, 55.14157828038837, 53.01419324341059, 54.12151433519213, 54.72321304758704], [64.0217433334945, 58.59624615156186, 64.37034195590547, 61.74080618135381, 67.52308363114038, 64.95730590909109, 44.11359964060699, 60.60376213192809, 62.6503191185711, 64.14971027238595, 61.85439770355395, 60.37766863378029]]
a4full_bp=[['PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON', 'PRON'], ['NOUN', 'ADJ', 'NOUN', 'NOUN', 'ADJ', 'VERB', 'ADJ', 'ADJ', 'NOUN', 'ADJ', 'NOUN', 'ADJ'], ['DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET', 'DET']]