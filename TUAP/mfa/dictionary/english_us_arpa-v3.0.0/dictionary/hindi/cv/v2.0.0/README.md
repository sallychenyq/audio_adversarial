
# Hindi CV dictionary v2.0.0

[Link to documentation on mfa-models](https://mfa-models.readthedocs.io/en/main/dictionary/hindi_cv.html)

Jump to section:

- [Dictionary details](#dictionary-details)
- [Intended use](#intended-use)
- [Performance Factors](#performance-factors)
- [Ethical considerations](#ethical-considerations)

## Dictionary details

- **Maintainer:** [Vox Communis](https://osf.io/t957v/)
- **Language:** [Hindi](https://en.wikipedia.org/wiki/Hindi_language)
- **Dialect:** N/A
- **Phone set:** [Epitran](https://github.com/dmort27/epitran)
- **Number of words:** `11,113`
- **Phones:** `a aː b b̤ d d̤ d͡ʒ d͡ʒ̤ e f h i iː j k kʰ l m n o p pʰ q r r̩ s t tʰ t͡ʃ t͡ʃʰ u uː v w x z ã ãː æ æː ŋ ũ ũː ɑ ɔ ɔː ɖ ɖ̤ ə ə̃ ɛ ɡ ɡ̤ ɣ ɦ ɪ ɲ ɳ ɹ ɹ̩ ɽ ɽ̥ ʂ ʃ ʈ ʈʰ ʊ ʌ θ ẽ`
- **License:** [CC-0](https://creativecommons.org/publicdomain/zero/1.0/)
- **Compatible MFA version:** `v2.0.0`
- **Citation:**

```bibtex
@misc{Ahn_Chodroff_2022,
	author={Ahn, Emily and Chodroff, Eleanor},
	title={VoxCommunis Corpus},
	address={\url{https://osf.io/t957v}},
	publisher={OSF},
	year={2022},
	month={Jan}
}
```

- If you have comments or questions about this dictionary or its phone set, you can check [previous MFA model discussion posts](https://github.com/MontrealCorpusTools/mfa-models/discussions?discussions_q=Hindi+CV+dictionary+v2.0.0) or create [a new one](https://github.com/MontrealCorpusTools/mfa-models/discussions/new).

## Installation

Install from the [MFA command line](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/models/index.html):

```
mfa model download dictionary hindi_cv
```

Or download from [the release page](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/dictionary-hindi_cv-v2.0.0).

## Intended use

This dictionary is intended for forced alignment of [Hindi](https://en.wikipedia.org/wiki/Hindi_language) transcripts.

This dictionary uses the [Epitran](https://github.com/dmort27/epitran) phone set for Hindi, and was used in training the Hindi [Epitran](https://github.com/dmort27/epitran) acoustic model. Pronunciations can be added on top of the dictionary, as long as no additional phones are introduced.

## Performance Factors

When trying to get better alignment accuracy, adding pronunciations is generally helpful, especially for different styles and dialects. The most impactful improvements will generally be seen when adding reduced variants that involve deleting segments/syllables common in spontaneous speech.  Alignment must include all phones specified in the pronunciation of a word, and each phone has a minimum duration (by default 10ms). If a speaker pronounces a multisyllabic word with just a single syllable, it can be hard for MFA to fit all the segments in, so it will lead to alignment errors on adjacent words as well.

## Ethical considerations

Deploying any Speech-to-Text model into any production setting has ethical implications. You should consider these implications before use.

### Demographic Bias

You should assume every machine learning model has demographic bias unless proven otherwise. For pronunciation dictionaries, it is often the case that transcription accuracy and lexicon coverage for the prestige variety modeled in this dictionary compared to other variants. If you are using this dictionary in production, you should acknowledge this as a potential issue.
