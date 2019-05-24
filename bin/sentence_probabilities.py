"""
Compute the probabilities for sentences by the language model used to train ELMO embeddings.

This is very similar to run_test and should probably be refactored together with it.
"""

import argparse
import logging
import sys

from bilm.training import test, load_options_latest_checkpoint, load_vocab, sentence_probabilities
from bilm.data import LMDataset, BidirectionalLMDataset
from vistautils.parameters import YAMLParametersLoader

from vistautils.parameters_only_entrypoint import parameters_only_entry_point


def main(params):
    options, ckpt_file = load_options_latest_checkpoint(params.existing_directory("save_dir"))

    logging.getLogger().setLevel(logging.INFO)

    # load the vocab
    #if 'char_cnn' in options:
    #    max_word_length = options['char_cnn']['max_characters_per_token']
    #else:
    #    max_word_length = None
    # vocab = load_vocab(args.vocab_file, max_word_length)

    sentence_file = params.existing_file("test_prefix")

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    # if options.get('bidirectional'):
    #     data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    # else:
    #     data = LMDataset(test_prefix, vocab, **kwargs)

#    sentence_probabilities(options, ckpt_file, data, batch_size=args.batch_size)
    burn_in_text_path = params.optional_existing_file("burn_in_text")

    if burn_in_text_path:
        with open(burn_in_text_path) as burn_in_text_inp:
            burn_in_text = list(burn_in_text_inp)
        print("Got burn in text")
    else:
        burn_in_text = None

    sentence_probabilities(options, ckpt_file, sentence_file,
                           params.existing_file("vocab_file"),
                           batch_size=params.positive_integer("batch_size"),
                           burn_in_text=burn_in_text,
                           bidirectional_losses=params.boolean("bidirectional_losses"))


if __name__ == '__main__':
    main(YAMLParametersLoader(interpolate_environmental_variables=True).load(sys.argv[1]))

