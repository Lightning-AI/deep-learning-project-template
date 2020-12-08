from project.lit_mnist import LitClassifier


# any function defined here will be able to load a model like:
# torch.hub.load("username/repo", "lit_classifier", *args, **kwargs)
# can put logic here for loading pretrained weights
def lit_classifier(*args, **kwargs):
    return LitClassifier(*args, **kwargs)
