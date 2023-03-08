from src.wrappers import Seq2SeqDM, TokenVocabulary
from src.model.model import TextTranslationTransformer
from src.model.model import VanillaTransformer
from src.translation.translators import TranslationInferenceGreedy, TranslationInferenceBeamSearch

available_models = {
    "vanilla": VanillaTransformer
}
