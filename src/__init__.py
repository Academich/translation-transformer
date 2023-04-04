from src.lightning_module import TextTranslationTransformer
from src.wrappers import CopySequence, Generic
from src.model.model import VanillaTransformer


model_catalogue = {
    "vanilla": VanillaTransformer
}
