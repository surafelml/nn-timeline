Progress on structuring project [reclaiming 2016]:

Best comeback that also claims the name sit_sml: 
  - 1 text (mt, lm), 1 vision (tracking), and 1 speech (st/asr) implementation and sample experiments and recipes.
  - aligns with multimodal MT 2022-23 tasks.

  - `Archs`: contains main skeletons of the specific arch (can be for diff tasks) being defined in Archs/ABC*. 
Other re-usable modules go under layers, etc. Here lets define `tnn` as arch and its components as a `template`.

  - Encoder 
    - add: layers/shared/dropout_layer, PositionalEmbedding, 
  - Decoder
    - add: 

Data
- Dictionary: called models/encoder_decoder_multi

Config
- TransformerConfig: and associated methods/classes 
- TODO: read about py dataclasses

Quant:
- see [quant diff](https://github.com/pytorch/fairseq/commit/1c8ab79ca59b466120e3df448673cab840f571ea#diff-f0eb88613a42eb9195c95be7d629fcabaeb7d451f8fe16587de03c433eb300d9)
to remove its introduction.

## Current step /generic setup/:
* Goal is to have the smallest code footprint possible - make it working in comparable way - focus on 
what matters (internal model details & adding new features & writing and teaching it.).

- I. FIRST ROUND SETUP
- rename internal sitMT -- src - keep sitMT as in python library layout - find proper name ?
- generator - done
- preprocess - done
- config (using dataclass, yaml, cli - smart decision - BLOG) - wip using https://github.com/omry/omegaconf
  - out of control with so much nonsense - just have 1 config.py file.
  - TODO: opting for OpenNMT-tf/Sockeye YAML based load_config in configy.py
  - from argparse import Namespace  # FairSeq depreciated Namespace and auto converting to OmegaConf - which is wise to follow ?
- search - done
- trainer - done
  - distributed training (using torch.distributed https://fairseq.readthedocs.io/en/latest/getting_started.html#distributed-training )  - removed.
    - 
  - once fit/skilled - try the implementation along with sharding large data.
  - optim - removing fp16 (as it requires its own large code)
- train - done
- translate - done
- done - `fixing errors in each file before final review for exp. decision: no half-precision, yes omega conf, not sure using DDP?`
- Reading before adding RNN network https://explained.ai/rnn/index.html 


- HERE: remember why going the hard way (top-to-details) - it's a lasting solution.
-- `fixing (structure, strong/week warnings) and cleaning before git and first BRANCH`. - DONE
-- `keep commenting out & adding clean note - don't remove`.
-- TODO: standardized import with comment
-- TODO: start experiment with initial version once the config is setup - then start cleaning up and simplification
-- sample for conversion args to omegaconf https://github.com/facebookresearch/fairseq/commit/a6a63279422f846a3c2f6c45b9c96d6951cc4b82 
--* after setting up v1.0 [branch out and build v0.1] the smallest/simplified possible implementation. 
--* by end of this proj. main question is why not implement it from bottom up, why top down and waste time?

- II. START FINAL CLEANUP & ORG BEFORE TEST EXP: TODO: 
  - i. arguments - [training approaches]() - dataclass - config - using python library structure than module or package
  - ii. module - loading

- III. START EXP AND FINAL COMMENTING ON EACH LINE TODO: 
  - i. model 
  - ii. ?
- IV. ADD NEW FEATURES/MONTHLY - WRITE ML BLOG
  - POSITION EMBEDS


---
# sit_ML
   - Make it succinct 
   - Human info/communication channels: speech-visual/image-text.
   - Reference tool for tutorials, self-experiments, study, etc. 
   - __Note__:
        - Approach to review fairseq/opennmt with K's code base is quite expensive, however, once the NMT model is done the rest is easy to navigate.
        - Priority is baseline NMT and ASR.
        - Build backward (from task > model > data > etc)
   - __TODO__
     - Impl. Blog: along with detail comments and unittest - write blog on specific modules (e.g. how pos-emb works ?)
       - write similar to `Annotated Transformer` as note book - describing each part.
       - can this reach for workshop/s or use sockeye (run both google/aws cloud) - to create surafelml/workshops/??
     - What really change/contribution? except changing the code base and simplification ? new design for other tasks?


### Why and What
- Concise: as a late comer to fairseq based seq2seq which by far has a large code base, 
I find myself lost or spending alot of time to achieve a minor task. 
This is why the need to replicate (for now the most used part - transformer seq2seq MT), while studying each code block.
, in this repo. As much as possible I have added in-line comments, it might help others too. 
For any license and borrowed script see the original FairSeq implementation.   


- In the past I have been lost in several seq2seq libraries, took me several weeks to figure out even the basics. 
These past years, I have worked with Nematus, OpenNMT-py/tf, and recently with FairSeq. One thing I understood (which
acutally from the begining), is the need to reconstruct everything from scratch. In this way, I have the chance to 
understanding the underlying working principles. 
- child project done on extra time and boaring time from work
- focused on skeleton text, speech, and vision tasks.
- inspires from opennmt, and fairseq.
- allows to easily and incrementally add features that belongs to one of the above tasks
- Providing Line by Line comments
  - comp science (dsa), maths (prob, stat, calculus), and ML (model, archs, optim, learning, etc)

### Description and Flow of the implementation

Create table and describe it as key-value/description pair:
- task - builds the xyz task, including loading dict, data, model, criterion
- model - ...

Insert a figure that shows the overall structure and map
- are there tools or manually mapping ?


### Features
- Vision see https://github.com/zhiqwang/sightseq
- Speech see FairSeq itself and how to best simplify it.
- Text see ?
- Multimodal see ?/ene

### Tasks and Timeline
 
 - A __comparison__ of RNN, CNN, TNN on NMT, ASR, S2S, I2T, etc
    - includes Overleaf/Arxiv/Notebook tutorial for technical details
    - Quite interesting (short, figures) way of [presenting](https://jessicastringham.net/2018/12/30/conv-max-pool/) the core concept of a model. I will only add the maths.
    - Timeline: ~ __1 month__ for one task/ a model. 
    - Reference [see](https://arxiv.org/abs/1806.06957) and others. 

- Features to Add and to PR
    - [Issue to customize dict/embs, adapters](https://github.com/pytorch/fairseq/issues/1241)
    - [Coverage penalty - penalities in general](https://github.com/pytorch/fairseq/issues/3024)
    - [zst paper 1 and 2](ADD)

- Language modeling with exported model for demo app.

* *Notes goes to Notebook/MD and Overleaf*

### Future tasks
- Re-implement vanilla Transformer with TF in `archs/tnn_tf` while keep the rest scripts.
- A simple tiny TNN, FNN with specific tasks.
- S2S - [see](https://arxiv.org/abs/1904.06037)
- [RNNSearch]((Bahdanau et al., 2014))
- Moses phrase based visualized with neural model?
- Discriminator networks for purpose XYZ
- TUTORIAL/S
    - Explains the figure of the model (each box, arrow, abstraction) in relation with the code portion
which is easier for beginners to quickly start building
    - Separates other modules (dat processing, optimization, search, etc) in detail 
- TBA
- Transition yemiketelwe bota be addis re-implement

### Examples
- Tiny Architecture for NMT, ?: `tiny_architecture`
- BirdLang-Am, Geeze - writing tutorial as getting started.
- Speech conversion from standard to BirdAmh
- Continuous improvement on Amh<>En (hand picked as hobby) 
- Speechesh (https://github.com/qute012/Wav2Keyword)
- Etc ... 


### References

- Initial [Transformer NMT impl.](https://colab.research.google.com/drive/1XXveSO76axz6hdZUGnr8wZqRsIUOdXSU#scrollTo=5S2BcIoOUgRg) 
- ?

---
### License 
Code is mostly ported from FairSeq implmentation, head over to ... for license. 
For the scripts that's not change at all you should see the original license at the header.

This repo is license under open source 


### Errors and Lessons
- changing namings at the begining instead of the end task after disecting out the transformer only part
- 