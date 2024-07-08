Documents are far too large to fit into spaCy on a reasonable GPU - some have length of 2,400,360; spaCy has a maximum of 1mil due to parsers requiring 1GB of memory per 100,000 characters.
I'm not using a parser, but it'll still make the transformer run too large

GPU RAM requirements mean I'm downsizing the training process as much as I can. Might include chunking in spaCy's preprocessing step. If I do chunk, then during inference I'll need to split text, run the model on each one, and aggregate results.
