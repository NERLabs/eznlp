from eznlp.io import ConllIO
from eznlp.utils import ChunksTagsTranslator

io = ConllIO(text_col_id=0, tag_col_id=1, scheme='BMES', encoding='utf-8', token_sep='', pad_token='<pad>')
data = io.read('data/HZ/hz_train.bmes')

print('Chunks:', data[0]['chunks'][:5])
print('Tokens 长度:', len(data[0]['tokens']))
print('\nTokens[0:10]:', data[0]['tokens'].raw_text[:10])
print('Tokens[4:7]:', data[0]['tokens'].raw_text[4:7], '(应该是: 坛坛枣)')
print('Tokens[19:22]:', data[0]['tokens'].raw_text[19:22])

# 直接看 BMES 文件内容
with open('data/HZ/hz_train.bmes', 'r') as f:
    lines = [l.strip() for l in f.readlines()[:50]]
    print('\n原始文件前50行:')
    for i, line in enumerate(lines[:10]):
        print(f'{i}: {line}')

translator = ChunksTagsTranslator(scheme='BMES')

# 检查是否有重叠
chunks = data[0]['chunks']
print(f'\n总共 {len(chunks)} 个 chunks')
for i, ck in enumerate(chunks[:10]):
    print(f'  {i}: {ck}')

# 按长度排序（和 chunks2tags 一样）
sorted_chunks = sorted(chunks, key=lambda ck: ck[2] - ck[1], reverse=True)
print(f'\n按长度排序后前5个:')
for ck in sorted_chunks[:5]:
    print(f'  {ck}, 长度={ck[2]-ck[1]}')

tags = translator.chunks2tags(data[0]['chunks'], len(data[0]['tokens']))
print('\nTags[4:7]:', tags[4:7], '(应该是 B-CUL, M-CUL, E-CUL)')
print('所有唯一标签:', set(tags))
