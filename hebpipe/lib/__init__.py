try:
    from .tt2conll import  conllize
    from .reorder_sgml import reorder
    from .dropout import WordDropout,LockedDropout
    from .crfutils.crf import CRF
    from .crfutils.viterbi import ViterbiDecoder,ViterbiLoss
except ModuleNotFoundError:
    from lib.tt2conll import  conllize
    from lib.reorder_sgml import reorder
    from lib.dropout import WordDropout,LockedDropout
    from lib.crfutils.crf import CRF
    from lib.crfutils.viterbi import ViterbiDecoder,ViterbiLoss

