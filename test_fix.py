#!/usr/bin/env python
from eznlp.utils import ChunksTagsTranslator

t = ChunksTagsTranslator(scheme='BMES')
tags = t.chunks2tags([('CUL', 4, 7), ('GEO', 19, 22)], 25)
print('Tags:', tags)
print('Tag[4:7]:', tags[4:7])
print('Tag[19:22]:', tags[19:22])
